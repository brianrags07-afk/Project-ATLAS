
import json
from pathlib import Path

from atlas.questions.question_library import build_core_question_library
from atlas.questions.question_expander import expand_question_library


QUESTION_ENGINE_VERSION = "1.1.0"


def run_question_engine(output_dir=None):
    if output_dir is None:
        output_dir = Path("/content/drive/MyDrive/Project_Atlas/data/questions")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    core_questions = build_core_question_library()
    expanded_questions = expand_question_library(core_questions, include_core=True)

    core_path = output_dir / "core_question_library.json"
    expanded_path = output_dir / "expanded_question_library.json"

    with open(core_path, "w") as f:
        json.dump(core_questions, f, indent=2)

    with open(expanded_path, "w") as f:
        json.dump(expanded_questions, f, indent=2)

    print("=" * 60)
    print("ATLAS QUESTION ENGINE")
    print("=" * 60)
    print(f"Core Questions..... {len(core_questions)}")
    print(f"Expanded Questions. {len(expanded_questions)}")
    print(f"Saved To........... {output_dir}")
    print("=" * 60)

    return {
        "engine": "ATLAS Question Engine",
        "engine_version": QUESTION_ENGINE_VERSION,
        "core_questions": len(core_questions),
        "expanded_questions": len(expanded_questions),
        "output_directory": str(output_dir),
    }
