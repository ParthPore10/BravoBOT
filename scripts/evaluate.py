import json
import os
from pathlib import Path

from app.rag_pipeline import answer_query

EVAL_PATH = Path("data/eval/questions.json")
OUTPUT_PATH = Path("data/eval/results.json")

def load_eval_questions():
    if not EVAL_PATH.exists():
        raise FileNotFoundError(
            f"Eval file not found at {EVAL_PATH}"
        )

    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    return questions

def score_pages(sources, expected_pages: list[int], answerable: bool):
    retrieved_pages = []

    for source in sources:
        doc = source.get("doc")
        chunk = source.get("chunk")

        if doc is not None:
            metadata = doc.metadata
        else:
            metadata = chunk

        page_number = metadata.get("page_number")

        if page_number is not None:
            retrieved_pages.append(int(page_number))

    retrieved_page_set = set(retrieved_pages)
    expected_page_set = set(expected_pages)

    if not answerable:
        passed = len(expected_page_set) == 0
        page_hits = []
        page_misses = []
        page_score = 1.0 if passed else 0.0

        return {
            "retrieved_pages": retrieved_pages,
            "page_hits": page_hits,
            "page_misses": page_misses,
            "page_score": page_score,
            "passed": passed,
        }

    page_hits = sorted(list(retrieved_page_set.intersection(expected_page_set)))
    page_misses = sorted(list(expected_page_set.difference(retrieved_page_set)))

    if expected_pages:
        page_score = len(page_hits) / len(expected_page_set)
    else:
        page_score = 0.0

    passed = len(page_hits) > 0

    return {
        "retrieved_pages": retrieved_pages,
        "page_hits": page_hits,
        "page_misses": page_misses,
        "page_score": page_score,
        "passed": passed,
    }

def evaluate():
    questions = load_eval_questions()
    results = []

    for index, item in enumerate(questions, start=1):
        question = item["question"]
        expected_pages = item.get("expected_pages", [])
        answerable = item.get("answerable", True)

        print(f"\n[{index}/{len(questions)}] Evaluating: {question}")

        response = answer_query(
            query=question,
            user_id=os.getenv("EVAL_USER_ID", "user-a"),
            candidate_k=5,
            final_k=3
        )

        answer = response["answer"]
        sources = response["sources"]

        score_result = score_pages(
            sources=sources,
            expected_pages=expected_pages,
            answerable=answerable
        )

        result = {
            "question": question,
            "answerable": answerable,
            "expected_pages": expected_pages,
            "answer": answer,
            "retrieved_pages": score_result["retrieved_pages"],
            "page_hits": score_result["page_hits"],
            "page_misses": score_result["page_misses"],
            "page_score": score_result["page_score"],
            "passed": score_result["passed"],
        }

        results.append(result)

        print(f"Retrieved Pages: {score_result['retrieved_pages']}")
        print(f"Page Hits: {score_result['page_hits']}")
        print(f"Page Score: {score_result['page_score']:.2f}")
        print(f"Passed: {score_result['passed']}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\nSaved evaluation results to {OUTPUT_PATH}")

if __name__ == "__main__":
    evaluate()
