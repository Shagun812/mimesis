import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset

from . import config



@dataclass(frozen=True)
class Question:
    """
    Canonical representation of one MIMESIS WMDP-Bio question.
    """

    question_id: str
    question: str
    choices: list[str]
    answer_index: int

    @property
    def answer_label(self) -> str:
        """Return the correct answer label (A/B/C/D)."""

        labels = ["A", "B", "C", "D"]

        if not 0 <= self.answer_index < len(labels):
            raise ValueError(
                f"Invalid answer index: {self.answer_index}"
            )

        return labels[self.answer_index]



def load_wmdp_bio():
    """
    Load the frozen WMDP-Bio source dataset from Hugging Face.

    MIMESIS expects:
        dataset: cais/wmdp
        configuration: wmdp-bio
        split: test
    """

    dataset = load_dataset(
        config.DATASET_NAME,
        config.DATASET_DOMAIN,
    )

    return dataset


def validate_dataset_schema(dataset) -> None:
    """
    Validate that the downloaded dataset matches the expected
    WMDP-Bio structure before normalization.
    """

    if "test" not in dataset:
        raise ValueError(
            "WMDP-Bio must contain a 'test' split."
        )

    expected_fields = {
        "question",
        "choices",
        "answer",
    }

    actual_fields = set(
        dataset["test"].features.keys()
    )

    missing = expected_fields - actual_fields

    if missing:
        raise ValueError(
            "WMDP-Bio test split is missing expected fields: "
            f"{sorted(missing)}"
        )


def inspect_dataset(dataset) -> dict[str, Any]:
    """
    Return basic information about the loaded dataset.
    """

    return {
        "splits": list(dataset.keys()),
        "num_rows": {
            split: len(dataset[split])
            for split in dataset.keys()
        },
        "features": {
            split: str(dataset[split].features)
            for split in dataset.keys()
        },
    }



def normalize_item(
    item: dict[str, Any],
    index: int,
) -> Question:
    """
    Convert one raw WMDP-Bio item into the canonical MIMESIS format.

    Expected raw fields:
        question
        choices
        answer
    """

    required_fields = {
        "question",
        "choices",
        "answer",
    }

    missing = required_fields - set(item.keys())

    if missing:
        raise KeyError(
            f"Question {index} is missing fields: "
            f"{sorted(missing)}"
        )

    question = str(item["question"]).strip()
    choices = [str(choice).strip() for choice in item["choices"]]
    answer = int(item["answer"])

    if not question:
        raise ValueError(
            f"Question {index} has empty question text."
        )

    if len(choices) != 4:
        raise ValueError(
            f"Question {index} has {len(choices)} choices; "
            "MIMESIS requires exactly four."
        )

    if any(not choice for choice in choices):
        raise ValueError(
            f"Question {index} contains an empty choice."
        )

    if not 0 <= answer < 4:
        raise ValueError(
            f"Question {index} has invalid answer index: {answer}."
        )

    return Question(
        question_id=f"wmdp_bio_{index:04d}",
        question=question,
        choices=choices,
        answer_index=answer,
    )


def normalize_split(split) -> list[Question]:
    """
    Normalize every item in a dataset split.
    """

    questions = [
        normalize_item(item, index)
        for index, item in enumerate(split)
    ]

    validate_questions(questions)

    return questions


# ---------------------------------------------------------------------------
# CANDIDATE POOL
# ---------------------------------------------------------------------------

def build_candidate_pool(
    output_path: Path = config.CANDIDATE_DATA_PATH,
) -> list[Question]:
    """
    Download WMDP-Bio, normalize its test split, validate it,
    and save the complete candidate pool.

    This is the pre-calibration candidate pool.

    It is NOT the experimental working set.
    """

    dataset = load_wmdp_bio()

    validate_dataset_schema(dataset)

    questions = normalize_split(
        dataset["test"]
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_questions(
        questions,
        output_path,
    )

    return questions



def is_eligible(
    honest_correct: bool,
    sandbagging_correct: bool,
) -> bool:
    """
    Apply the primary MIMESIS eligibility rule.

    Eligible:
        Honest      -> correct
        Sandbagging -> incorrect
    """

    return (
        honest_correct
        and not sandbagging_correct
    )


def validate_behavioral_results(
    questions: list[Question],
    honest_results: dict[str, bool],
    sandbagging_results: dict[str, bool],
) -> None:
    """
    Ensure behavioral results exist for every candidate question.

    Missing results are errors because silently dropping questions
    would make the working-set construction non-reproducible.
    """

    expected_ids = {
        question.question_id
        for question in questions
    }

    honest_ids = set(honest_results.keys())
    sandbagging_ids = set(sandbagging_results.keys())

    missing_honest = expected_ids - honest_ids
    missing_sandbagging = expected_ids - sandbagging_ids

    if missing_honest:
        raise KeyError(
            "Missing Honest results for: "
            f"{sorted(missing_honest)}"
        )

    if missing_sandbagging:
        raise KeyError(
            "Missing Sandbagging results for: "
            f"{sorted(missing_sandbagging)}"
        )


def filter_working_set(
    questions: list[Question],
    honest_results: dict[str, bool],
    sandbagging_results: dict[str, bool],
) -> list[Question]:
    """
    Construct the MIMESIS working set from behavioral results.

    Primary eligibility criterion:

        Honest      = correct
        Sandbagging = incorrect
    """

    validate_behavioral_results(
        questions=questions,
        honest_results=honest_results,
        sandbagging_results=sandbagging_results,
    )

    working_set = [
        question
        for question in questions
        if is_eligible(
            honest_correct=honest_results[
                question.question_id
            ],
            sandbagging_correct=sandbagging_results[
                question.question_id
            ],
        )
    ]

    return working_set



def write_questions(
    questions: list[Question],
    path: Path,
) -> None:
    """
    Write canonical questions as JSONL.
    """

    validate_questions(questions)

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for question in questions:
            file.write(
                json.dumps(
                    asdict(question),
                    ensure_ascii=False,
                )
                + "\n"
            )


def read_questions(
    path: Path,
) -> list[Question]:
    """
    Read canonical questions from JSONL and validate them.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Question file does not exist: {path}"
        )

    questions = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                item = json.loads(line)

                question = Question(
                    question_id=item["question_id"],
                    question=str(item["question"]),
                    choices=[
                        str(choice)
                        for choice in item["choices"]
                    ],
                    answer_index=int(
                        item["answer_index"]
                    ),
                )

                validate_question(question)

                questions.append(question)

            except (
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
            ) as exc:

                raise ValueError(
                    f"Invalid question on line "
                    f"{line_number}: {exc}"
                ) from exc

    validate_questions(questions)

    return questions



def save_working_set(
    questions: list[Question],
    path: Path = config.WORKING_SET_PATH,
) -> None:
    """
    Save the selected MIMESIS working set.

    The caller is responsible for freezing the behavioral
    results and recording the selection in freeze_log.md.
    """

    if not questions:
        raise ValueError(
            "Cannot save an empty MIMESIS working set."
        )

    if len(questions) > config.TARGET_WORKING_SET_SIZE:
        raise ValueError(
            f"Working set contains {len(questions)} questions, "
            f"exceeding target size "
            f"{config.TARGET_WORKING_SET_SIZE}."
        )

    write_questions(
        questions,
        path,
    )


def load_working_set(
    path: Path = config.WORKING_SET_PATH,
) -> list[Question]:
    """
    Load and validate the frozen MIMESIS working set.
    """

    questions = read_questions(path)

    if len(questions) < config.MIN_WORKING_SET_SIZE:
        raise ValueError(
            f"Working set contains only {len(questions)} questions; "
            f"MIMESIS requires at least "
            f"{config.MIN_WORKING_SET_SIZE}."
        )

    if len(questions) > config.TARGET_WORKING_SET_SIZE:
        raise ValueError(
            f"Working set contains {len(questions)} questions; "
            f"maximum target is "
            f"{config.TARGET_WORKING_SET_SIZE}."
        )

    return questions




def validate_question(
    question: Question,
) -> None:
    """
    Validate one canonical question.
    """

    if not isinstance(question, Question):
        raise TypeError(
            "Expected a Question instance."
        )

    if not question.question_id.strip():
        raise ValueError(
            "Question ID cannot be empty."
        )

    if not question.question.strip():
        raise ValueError(
            f"{question.question_id}: empty question."
        )

    if len(question.choices) != 4:
        raise ValueError(
            f"{question.question_id}: "
            "expected exactly four choices."
        )

    if any(
        not isinstance(choice, str)
        or not choice.strip()
        for choice in question.choices
    ):
        raise ValueError(
            f"{question.question_id}: "
            "all choices must be non-empty strings."
        )

    if not 0 <= question.answer_index < 4:
        raise ValueError(
            f"{question.question_id}: "
            "invalid answer index."
        )


def validate_questions(
    questions: list[Question],
) -> None:
    """
    Validate an entire question collection.
    """

    if not questions:
        raise ValueError(
            "Question collection is empty."
        )

    seen_ids: set[str] = set()

    for question in questions:

        validate_question(question)

        if question.question_id in seen_ids:
            raise ValueError(
                f"Duplicate question ID: "
                f"{question.question_id}"
            )

        seen_ids.add(question.question_id)


if __name__ == "__main__":

    config.validate_config()

    print("Loading WMDP-Bio...")

    dataset = load_wmdp_bio()

    validate_dataset_schema(dataset)

    print("\nDataset:")
    print(inspect_dataset(dataset))

    questions = normalize_split(
        dataset["test"]
    )

    print(
        f"\nNormalized {len(questions)} "
        "WMDP-Bio questions."
    )

    output_path = config.CANDIDATE_DATA_PATH

    write_questions(
        questions,
        output_path,
    )

    print(
        f"Candidate pool written to: "
        f"{output_path}"
    )