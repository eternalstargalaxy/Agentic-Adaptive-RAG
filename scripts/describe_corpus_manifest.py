from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graph.corpus_pipeline import describe_corpus_manifest
from graph.corpus_profiles import CORPUS_PROFILES


def main() -> None:
    parser = argparse.ArgumentParser(description="Describe the versioned corpus manifest.")
    parser.add_argument("--profile", default="medical_demo")
    args = parser.parse_args()

    profile = CORPUS_PROFILES[args.profile]
    payload = describe_corpus_manifest(profile)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
