"""Run a model the way each evaluated system uses it.

Three systems are compared on the same held-out questions:

* ``finetuned`` base model + LoRA adapter, prompted exactly as in training
* ``base``      the untuned base model, given an instruction listing the allowed values
* ``rag``       the untuned base model, given the instruction *and* the k most similar
                training examples (BM25) as worked examples

The baselines get the format that works best for them. On the example model
(SmolLM2-135M-Instruct, held-out exact match) the raw few-shot format scored 6.9 % for
RAG against 1.4 % through the chat template, so ``raw`` is the default; ``chat`` stays
available (``[eval] baseline_format``) for larger instruction-tuned models. A baseline
handicapped by a format it does poorly with would make fine-tuning look better than it
is.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lineage.evaluate.retrieval import BM25
from lineage.task import TaskSpec


def instruction(task: TaskSpec) -> str:
    """The instruction given to the untuned systems."""
    lines = ["Triage the IT support ticket. Answer with exactly these lines and nothing else:"]
    for name, allowed in task.fields.items():
        lines.append(f"{name}: one of {', '.join(allowed)}")
    return "\n".join(lines)


@dataclass
class Model:
    """A loaded model + tokenizer, ready to generate and score."""

    model: Any
    tokenizer: Any
    name: str

    @classmethod
    def load(cls, snapshot: Path, adapter: Path | None, name: str, threads: int = 4) -> Model:
        """Load the base model, merging an adapter if given."""
        import torch

        from lineage.train.lora import load_base

        torch.set_num_threads(threads)
        model, tokenizer = load_base(snapshot)
        if adapter is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
        model.eval()
        tokenizer.padding_side = "left"
        return cls(model, tokenizer, name)

    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 16,
        batch: int = 16,
        stop: list[str] | None = None,
    ) -> list[str]:
        """Greedy completions (deterministic), returned in the order of ``prompts``.

        Prompts are grouped by length before batching so that a long prompt (a RAG
        prompt with four worked examples) does not pad a whole batch of short ones.
        """
        import torch

        lengths = [len(self.tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts]
        order = sorted(range(len(prompts)), key=lambda i: lengths[i])
        outputs: list[str] = [""] * len(prompts)
        for start in range(0, len(order), batch):
            indices = order[start : start + batch]
            chunk = [prompts[i] for i in indices]
            enc = self.tokenizer(chunk, return_tensors="pt", padding=True, add_special_tokens=False)
            with torch.no_grad():
                out = self.model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    stopping_criteria=_stop_on(self.tokenizer, stop, enc["input_ids"].shape[1])
                    if stop
                    else None,
                )
            new = out[:, enc["input_ids"].shape[1] :]
            for i, text in zip(
                indices, self.tokenizer.batch_decode(new, skip_special_tokens=True), strict=True
            ):
                outputs[i] = text
        return outputs

    def logprob(self, context: str, continuation: str) -> float:
        """Sum of log-probabilities of ``continuation`` tokens after ``context``."""
        import torch

        ctx = self.tokenizer(context, add_special_tokens=False)["input_ids"]
        cont = self.tokenizer(continuation, add_special_tokens=False)["input_ids"]
        ids = torch.tensor([ctx + cont])
        with torch.no_grad():
            logits = self.model(input_ids=ids).logits[0, :-1]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        positions = range(len(ctx) - 1, len(ctx) + len(cont) - 1)
        return float(sum(logprobs[p, ids[0, p + 1]] for p in positions))

    def logprobs(self, context: str, continuations: list[str], batch: int = 64) -> list[float]:
        """Batched :meth:`logprob` for many continuations of one shared context.

        The context is run once and its key/value cache reused for every batch of
        continuations: scoring 256 canary candidates costs one context pass plus the
        candidates' own tokens, instead of 256 full sequences.
        """
        import copy

        import torch

        ctx = self.tokenizer(context, add_special_tokens=False)["input_ids"]
        conts = [self.tokenizer(c, add_special_tokens=False)["input_ids"] for c in continuations]
        pad = self.tokenizer.pad_token_id
        with torch.no_grad():
            prefix = self.model(input_ids=torch.tensor([ctx]), use_cache=True)
        first = torch.log_softmax(prefix.logits[0, -1].float(), dim=-1)
        results: list[float] = []
        for start in range(0, len(conts), batch):
            chunk = conts[start : start + batch]
            width = max(len(c) for c in chunk)
            ids = torch.tensor([c + [pad] * (width - len(c)) for c in chunk])
            valid = torch.tensor([[1] * len(c) + [0] * (width - len(c)) for c in chunk])
            mask = torch.cat([torch.ones(len(chunk), len(ctx), dtype=valid.dtype), valid], dim=1)
            cache = copy.deepcopy(prefix.past_key_values)
            cache.batch_repeat_interleave(len(chunk))
            with torch.no_grad():
                logits = self.model(
                    input_ids=ids, attention_mask=mask, past_key_values=cache, use_cache=True
                ).logits
            logp = torch.log_softmax(logits.float(), dim=-1)
            # Token j of a continuation is predicted by position j - 1 of this batch;
            # token 0 by the last position of the context.
            following = logp[:, :-1].gather(2, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            following = (following * valid[:, 1:]).sum(dim=1)
            heads = first[ids[:, 0]]
            results += [float(v) for v in heads + following]
        return results


def _stop_on(tokenizer: Any, stop: list[str], prompt_length: int) -> Any:
    """Stop each sequence once its *generated* text contains one of ``stop``.

    transformers' own ``stop_strings`` relies on tokenizer internals that some
    tokenizers lack; decoding the new tokens works with any tokenizer.
    """
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    class StopOnText(StoppingCriteria):
        def __call__(self, input_ids: Any, _scores: Any, **_kwargs: Any) -> Any:
            texts = tokenizer.batch_decode(input_ids[:, prompt_length:], skip_special_tokens=True)
            return torch.tensor([any(m in t for m in stop) for t in texts], dtype=torch.bool)

    return StoppingCriteriaList([StopOnText()])


class Prompter:
    """Builds the prompt each system sees."""

    def __init__(
        self,
        task: TaskSpec,
        tokenizer: Any,
        train: list[tuple[str, str]],
        k: int = 4,
        baseline_format: str = "raw",
    ):
        self.task = task
        self.tokenizer = tokenizer
        self.train = train
        self.k = k
        self.index = BM25([x for x, _ in train]) if train else None
        self.chat = baseline_format == "chat" and bool(getattr(tokenizer, "chat_template", None))

    def finetuned(self, text: str) -> str:
        """Exactly the training prompt."""
        return self.task.prompt(text)

    def base(self, text: str) -> str:
        """Instruction + ticket."""
        if self.chat:
            return self._chat(f"{instruction(self.task)}\n\nTicket: {text}")
        return f"{instruction(self.task)}\n\n{self.task.prompt(text)}"

    def rag(self, text: str) -> str:
        """Instruction + retrieved worked examples + ticket."""
        assert self.index is not None
        shots = [self.train[i] for i in self.index.top(text, self.k)]
        if self.chat:
            examples = "\n\n".join(f"Ticket: {x}\n{y}" for x, y in shots)
            return self._chat(
                f"{instruction(self.task)}\n\nExamples:\n\n{examples}\n\nTicket: {text}"
            )
        examples = "\n".join(f"{self.task.prompt(x)}{y}\n" for x, y in shots)
        return f"{instruction(self.task)}\n\n{examples}{self.task.prompt(text)}"

    def _chat(self, content: str) -> str:
        rendered: str = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
        )
        return rendered


def first_answer(text: str) -> str:
    """Cut a completion at the point where a few-shot model starts the next example."""
    return text.split("###", 1)[0].strip()
