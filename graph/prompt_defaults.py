from __future__ import annotations

from textwrap import dedent

from graph.corpus_pipeline import describe_corpus_manifest
from graph.corpus_profiles import get_active_corpus_profile

DEFAULT_QUERY_TOWER_INSTRUCTION = (
    "Represent this medical query for retrieving clinically relevant evidence "
    "about symptoms, diagnoses, lab interpretation, medications, contraindications, "
    "guideline recommendations, and multi-condition management: "
)

MEDICAL_SAFETY_POLICY = dedent(
    """
    Medical safety principles:
    - Prefer evidence-backed general medical information over personalized diagnosis.
    - Do not invent medication doses, contraindications, interactions, or monitoring thresholds.
    - If the evidence is incomplete, say so explicitly instead of guessing.
    - For emergency warning signs such as severe breathing difficulty, confusion, or chest pain,
      it is acceptable to add a short urgent-care reminder.
    """
).strip()


def build_profile_prompt_context() -> str:
    profile = get_active_corpus_profile()
    manifest_summary = describe_corpus_manifest(profile)

    topic_lines = "\n".join(f"- {topic}" for topic in profile.routing_topics) or "- None"
    benchmark_lines = (
        "\n".join(f"- {metric}" for metric in profile.benchmark_focus) or "- None"
    )

    assets = manifest_summary.get("assets", [])
    asset_lines = (
        "\n".join(
            "- "
            f"{asset.get('name', 'unknown')} "
            f"(usage={asset.get('usage', 'unknown')}, kind={asset.get('kind', 'unknown')}): "
            f"{asset.get('description', '')}"
            for asset in assets
        )
        or "- No manifest assets declared."
    )

    return (
        f"Current corpus profile: {profile.display_name}\n"
        f"Profile description: {profile.description}\n\n"
        f"Main covered topics:\n{topic_lines}\n\n"
        f"Available corpus assets:\n{asset_lines}\n\n"
        f"Main benchmark focuses:\n{benchmark_lines}"
    )
