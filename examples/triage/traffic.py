"""Send demo traffic to the gateway: the held-out set, then shifted tickets."""

from __future__ import annotations

import json
import random
import sys
import urllib.request
from pathlib import Path


def post(base: str, text: str) -> dict:
    request = urllib.request.Request(
        f"http://{base}/v1/triage", data=json.dumps({"ticket": text}).encode()
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def drift(base: str) -> dict:
    with urllib.request.urlopen(f"http://{base}/drift", timeout=30) as response:
        return json.loads(response.read())


def main() -> None:
    base, heldout = sys.argv[1], sys.argv[2]
    rows = [json.loads(line) for line in Path(heldout).read_text(encoding="utf-8").splitlines()]
    print(post(base, rows[0]["input"]))
    for row in rows:
        post(base, row["input"])
    scores = drift(base)
    print(f"after {len(rows)} held-out tickets: breaches={scores['breaches']}")
    rng = random.Random(1)
    subjects = [
        "le portail RH n'affiche plus mes congés",
        "impossible de valider ma note de frais",
        "le planning des astreintes est vide",
        "ma fiche de paie est introuvable",
    ]
    for _ in range(130):
        post(base, f"Bonjour, {rng.choice(subjects)} depuis lundi. Merci de regarder.")
    scores = drift(base)
    print("after 130 shifted tickets:", json.dumps(scores["scores"]))
    for breach in scores["breaches"]:
        print(f"  ALERT {breach['signal']} {breach['score']} > {breach['threshold']}")


if __name__ == "__main__":
    main()
