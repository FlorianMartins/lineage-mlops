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

    def generate(self, prompts: list[str], max_new_tokens: int = 16, batch: int = 16) -> list[str]:
        """Greedy completions (deterministic)."""
        import torch

        outputs: list[str] = []
        for start in range(0, len(prompts), batch):
            chunk = prompts[start : start + batch]
            enc = self.tokenizer(chunk, return_tensors="pt", padding=True, add_special_tokens=False)
            with torch.no_grad():
                out = self.model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            new = out[:, enc["input_ids"].shape[1] :]
            outputs += self.tokenizer.batch_decode(new, skip_special_tokens=True)
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
        """Batched :meth:`logprob` for many continuations of one context."""
        import torch

        ctx = self.tokenizer(context, add_special_tokens=False)["input_ids"]
        conts = [self.tokenizer(c, add_special_tokens=False)["input_ids"] for c in continuations]
        pad = self.tokenizer.pad_token_id
        results: list[float] = []
        for start in range(0, len(conts), batch):
            chunk = conts[start : start + batch]
            width = len(ctx) + max(len(c) for c in chunk)
            rows = [ctx + c + [pad] * (width - len(ctx) - len(c)) for c in chunk]
            mask = [[1] * (len(ctx) + len(c)) + [0] * (width - len(ctx) - len(c)) for c in chunk]
            with torch.no_grad():
                logits = self.model(
                    input_ids=torch.tensor(rows), attention_mask=torch.tensor(mask)
                ).logits
            logp = torch.log_softmax(logits.float(), dim=-1)
            for row, cont in enumerate(chunk):
                total = 0.0
                for offset, token in enumerate(cont):
                    total += float(logp[row, len(ctx) + offset - 1, token])
                results.append(total)
        return results


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
