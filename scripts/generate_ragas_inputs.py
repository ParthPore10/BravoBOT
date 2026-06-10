import json
import os
from pathlib import Path

from app.rag_pipeline import answer_query, get_result_text

QUESTIONS_PATH = Path("data/eval/questions.json")
OUTPUT_PATH = Path("data/eval/ragas_inputs.json")


def main():
    questions = json.loads(
        QUESTIONS_PATH.read_text(encoding="utf-8")
    )
    records = []

    for index, item in enumerate(questions, start=1):
        print(f"[{index}/{len(questions)}] {item['question']}")

        result = answer_query(
            query=item["question"],
            user_id=os.getenv("EVAL_USER_ID", "user-a"),
            candidate_k=5,
            final_k=3,
        )

        contexts = [
            get_result_text(source)
            for source in result["sources"]
        ]

        records.append({
            "question_id": item["question_id"],
            "user_input": item["question"],
            "response": result["answer"],
            "retrieved_contexts": contexts,
            "reference": item["reference_answer"],
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved RAGAS inputs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
