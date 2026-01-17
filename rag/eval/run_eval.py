from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Dict, Any

from rag.answer import answer_question

QUESTIONS = [
    "What is the PRA’s supervisory approach?",
    "What are the PRA’s primary objectives?",
    "Explain CET1 in the context of Barclays.",
    "What is the Proactive Intervention Framework (PIF)?",
    "How is CET1 calculated step by step?",
]

def main() -> None:
    out_dir = Path("rag/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "eval_results.csv"

    rows: List[Dict[str, Any]] = []
    not_found = 0

    for q in QUESTIONS:
        res = answer_question(q, k=4, llm_model="llama3.2:3b")
        ans = res.get("answer", "")
        citations = res.get("citations", []) or []

        is_nf = ans.strip().startswith("Not found in the provided documents.")
        if is_nf:
            not_found += 1

        rows.append(
            {
                "question": q,
                "not_found": is_nf,
                "citations": " | ".join(citations),
                "answer_preview": ans.replace("\n", " ")[:220],
            }
        )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["question", "not_found", "citations", "answer_preview"])
        w.writeheader()
        w.writerows(rows)

    total = len(QUESTIONS)
    answered = total - not_found
    print(f"Eval done: {answered}/{total} answered, {not_found}/{total} not_found")
    print(f"Saved: {out_csv}")

if __name__ == "__main__":
    main()
