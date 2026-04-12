from __future__ import annotations

from typing import Any, Dict, List

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader


def _stringify_documents(documents: List[Any]) -> str:
    return "\n\n".join(
        getattr(document, "page_content", str(document))
        for document in documents
    )


def _safe_ragas_faithfulness(question: str, generation: str, documents: List[Any]) -> float | None:
    try:
        from ragas import SingleTurnSample
        from ragas.metrics import Faithfulness
    except Exception:
        return None

    try:
        sample = SingleTurnSample(
            user_input=question,
            response=generation,
            retrieved_contexts=[
                getattr(document, "page_content", str(document))
                for document in documents
            ],
        )
        scorer = Faithfulness()
        return float(scorer.single_turn_score(sample))
    except Exception:
        return None


def evaluate_generation(question: str, generation: str, documents: List[Any]) -> Dict[str, Any]:
    grounding_grade = hallucination_grader.invoke(
        {"documents": _stringify_documents(documents), "generation": generation}
    )
    answer_grade = answer_grader.invoke(
        {"question": question, "generation": generation}
    )
    faithfulness = _safe_ragas_faithfulness(question, generation, documents)

    grounded = bool(grounding_grade.binary_score)
    if faithfulness is not None:
        grounded = grounded and faithfulness >= 0.5

    return {
        "grounded": grounded,
        "addresses_question": bool(answer_grade.binary_score),
        "faithfulness_score": faithfulness,
    }
