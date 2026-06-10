import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from ragas.run_config import RunConfig

INPUT_PATH = Path("data/eval/ragas_inputs.json")
OUTPUT_PATH = Path("data/eval/ragas_results.json")

def main():
    load_dotenv()

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} does not exist. Run "
            "`python -m scripts.generate_ragas_inputs` first."
        )

    records = json.loads(INPUT_PATH.read_text(encoding="utf-8"))

    if not records:
        raise ValueError("RAGAS input file contains no records")

    # Use two questions initially. Set RAGAS_LIMIT=0 for all questions.
    limit = int(os.getenv("RAGAS_LIMIT", "1"))

    if limit > 0:
        records = records[:limit]

    samples = [
        SingleTurnSample(
            user_input=item["user_input"],
            response=item["response"],
            retrieved_contexts=item["retrieved_contexts"],
            reference=item["reference"],
        )
        for item in records
    ]

    dataset = EvaluationDataset(samples=samples)

    judge_llm = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "mistral:latest"),
        base_url="http://localhost:11434",
        temperature=0,
    )

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434",
    )

    run_config = RunConfig(
        timeout=600,
        max_retries=1,
        max_wait=10,
        max_workers=1,
    )

    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=judge_llm,
        embeddings=embeddings,
        run_config=run_config,
        raise_exceptions=True,
    )

    scored_records = []

    for item, scores in zip(records, result.scores):
        scored_records.append({
            **item,
            "scores": {
                name: float(value)
                for name, value in scores.items()
            },
        })

    metric_names = result.scores[0].keys()

    averages = {
        name: (
            sum(float(row[name]) for row in result.scores)
            / len(result.scores)
        )
        for name in metric_names
    }

    output = {
        "evaluated_questions": len(records),
        "averages": averages,
        "results": scored_records,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nAverage RAGAS scores:")

    for name, score in averages.items():
        print(f"{name}: {score:.4f}")

    print(f"\nSaved results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()