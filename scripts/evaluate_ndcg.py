import json
import math
import os
from pathlib import Path

from app.hybrid_retriever import hybrid_search
from app.rag_pipeline import generate_hyde_document

QUESTIONS_PATH = Path("data/eval/questions.json")
OUTPUT_PATH = Path("data/eval/ndcg_results.json")
k=5

def calculate_dcg(relevance_score: list[int], k:int)->float:
    score =0.0

    for rank,relevance in enumerate(relevance_score[:k], start=1):
        gain = (2**relevance)-1
        discount = math.log2(rank+1)
        score+= gain/discount
    return score

def calculate_ndcg(
        retrieved_chunk_ids:list[str],
        relevance :dict[str,int],
        k :int
)-> float:

    retrieved_score = [
        relevance.get(chunk_id,0)
        for chunk_id in retrieved_chunk_ids[:k]
    ]
    dcg = calculate_dcg(retrieved_score,k)

    ideal_score = sorted(
        relevance.values(),
        reverse=True
    )[:k]

    idcg = calculate_dcg(ideal_score, k)

    if idcg ==0:
        return 0.0

    return dcg/idcg


def main():
    questions = json.loads(
        QUESTIONS_PATH.read_text(encoding="utf-8")
    )

    results = []

    for index,item in enumerate(questions,start=1):
        relevance = item.get("relevance")

        if not relevance:
            print(
                f"[{index}/{len(questions)}] "
                f"Skipping {item['question_id']}: no relevance labels"
            )
            continue
        question = item["question"]

        print(
            f"[{index}/{len(questions)}] "
            f"Evaluating {item['question_id']}"
        )

        dense_query = generate_hyde_document(question)

        retrieved_results = hybrid_search(original_query=question,
                                          dense_query=dense_query,
                                          user_id=os.getenv(
                                              "EVAL_USER_ID",
                                              "user-a",
                                          ),
                                          top_k=k)

        retrieved_chunk_ids = [
            result['chunk_id']
            for result in retrieved_results
        ]

        relevance_scores = [
            relevance.get(chunk_id, 0)
            for chunk_id in retrieved_chunk_ids
        ]
        ndcg = calculate_ndcg(
                    retrieved_chunk_ids=retrieved_chunk_ids,
                    relevance=relevance,
                    k=k,
                )
        result = {
            "question_id": item["question_id"],
            "question": question,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "relevance_scores": relevance_scores,
            f"ndcg@{k}": ndcg,
        }

        results.append(result)

        print(f"Retrieved relevance: {relevance_scores}")
        print(f"NDCG@{k}: {ndcg:.4f}")

    if not results:
        raise ValueError(
            "No questions contained relevance labels"
        )

    metric_name = f"ndcg@{k}"
    average_ndcg = (
        sum(result[metric_name] for result in results)
        / len(results)
    )

    output = {
        "k": k,
        "evaluated_questions": len(results),
        "average_ndcg": average_ndcg,
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nAverage NDCG@{k}: {average_ndcg:.4f}")
    print(f"Saved results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
