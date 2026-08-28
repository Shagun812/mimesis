
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset

from . import config



#DATA STRUCTURE

@dataclass(frozen=True)
class Question:
    """Canonical representation of one MIMESIS question."""

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


#RAW DATASET LOADING

def load_wmdp_bio():
    """
    Load the WMDP-Bio dataset from Hugging Face.

    The exact dataset schema is inspected rather than assumed.
    """

    dataset = load_dataset(
        config.DATASET_NAME,
        "wmdp-bio",
    )

    return dataset


def inspect_dataset(dataset) -> dict[str, Any]:
    """Return basic information about the loaded dataset."""

    result = {
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

    return result



#SCHEMA NORMALIZATION

def normalize_item(
    item: dict[str, Any],
    index: int,
) -> Question:
    """
    Convert one raw WMDP item into the canonical MIMESIS format.

    WMDP multiple-choice items use:
        question
        choices
        answer
    """

    if "question" not in item:
        raise KeyError("Dataset item is missing 'question'.")

    if "choices" not in item:
        raise KeyError("Dataset item is missing 'choices'.")

    if "answer" not in item:
        raise KeyError("Dataset item is missing 'answer'.")

    question = str(item["question"]).strip()
    choices = list(item["choices"])
    answer = int(item["answer"])

    if not question:
        raise ValueError(
            f"Question {index} has empty question text."
        )

    if len(choices) != 4:
        raise ValueError(
            f"Question {index} has {len(choices)} choices; "
            "MIMESIS expects four."
        )

    if not 0 <= answer < len(choices):
        raise ValueError(
            f"Question {index} has invalid answer index: {answer}."
        )

    return Question(
        question_id=f"wmdp_bio_{index:04d}",
        question=question,
        choices=[str(choice) for choice in choices],
        answer_index=answer,
    )


def normalize_split(split) -> list[Question]:
    """Normalize every item in a dataset split."""

    return [
        normalize_item(item, index)
        for index, item in enumerate(split)
    ]


#CANDIDATE POOL

def build_candidate_pool(
    output_path: Path = config.CANDIDATE_DATA_PATH,
) -> list[Question]:
    """
    Load WMDP-Bio and save the candidate pool locally.

    This is the pre-calibration pool. It is NOT the frozen
    experimental working set.
    """

    dataset = load_wmdp_bio()

    # WMDP-Bio is expected to expose a test split.


    questions = normalize_split(dataset["train"])
    if "test" not in dataset:
        raise ValueError(
            "Expected a 'test' split in WMDP-Bio."
        )

questions = normalize_split(dataset["test"])
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_questions(
        questions,
        output_path,
    )

    return questions



#BEHAVIORAL ELIGIBILITY

def is_eligible(
    honest_correct: bool,
    sandbagging_correct: bool,
) -> bool:
    """
    Apply the primary MIMESIS working-set rule.

    Eligible:
        Honest      -> correct
        Sandbagging -> incorrect
    """

    return (
        honest_correct
        and not sandbagging_correct
    )


def filter_working_set(
    questions: list[Question],
    honest_results: dict[str, bool],
    sandbagging_results: dict[str, bool],
) -> list[Question]:
    """
    Construct the MIMESIS working set from behavioral results.

    The question must satisfy:
        Honest = correct
        Sandbagging = incorrect
    """

    working_set = []

    for question in questions:
        question_id = question.question_id

        if question_id not in honest_results:
            raise KeyError(
                f"Missing Honest result for {question_id}."
            )

        if question_id not in sandbagging_results:
            raise KeyError(
                f"Missing Sandbagging result for {question_id}."
            )

        if is_eligible(
            honest_correct=honest_results[question_id],
            sandbagging_correct=sandbagging_results[question_id],
        ):
            working_set.append(question)

    return working_set



#SERIALIZATION

def write_questions(
    questions: list[Question],
    path: Path,
) -> None:
    """Write questions as JSONL."""

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
    """Read canonical questions from JSONL."""

    if not path.exists():
        raise FileNotFoundError(
            f"Question file does not exist: {path}"
        )

    questions = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            item = json.loads(line)

            try:
                questions.append(
                    Question(
                        question_id=item["question_id"],
                        question=item["question"],
                        choices=list(item["choices"]),
                        answer_index=int(item["answer_index"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid question on line {line_number}: {exc}"
                ) from exc

    return questions



#WORKING-SET HELPERS

def save_working_set(
    questions: list[Question],
    path: Path = config.WORKING_SET_PATH,
) -> None:
    """Save the frozen MIMESIS working set."""

    write_questions(
        questions,
        path,
    )


def load_working_set(
    path: Path = config.WORKING_SET_PATH,
) -> list[Question]:
    """Load the frozen MIMESIS working set."""

    return read_questions(path)


#VALIDATION

def validate_question(question: Question) -> None:
    """Validate one canonical question."""

    if not question.question_id:
        raise ValueError(
            "Question ID cannot be empty."
        )

    if not question.question.strip():
        raise ValueError(
            f"{question.question_id}: empty question."
        )

    if len(question.choices) != 4:
        raise ValueError(
            f"{question.question_id}: expected four choices."
        )

    if not 0 <= question.answer_index < 4:
        raise ValueError(
            f"{question.question_id}: invalid answer index."
        )


def validate_questions(
    questions: list[Question],
) -> None:
    """Validate an entire question collection."""

    if not questions:
        raise ValueError(
            "Question collection is empty."
        )

    seen_ids = set()

    for question in questions:
        validate_question(question)

        if question.question_id in seen_ids:
            raise ValueError(
                f"Duplicate question ID: {question.question_id}"
            )

        seen_ids.add(question.question_id)



#MAIN

if __name__ == "__main__":
    config.validate_config()

    print("Loading WMDP-Bio...")

    dataset = load_wmdp_bio()

    print("\nDataset:")
    print(inspect_dataset(dataset))

    questions = normalize_split(dataset["train"])

    validate_questions(questions)

    print(
        f"\nNormalized {len(questions)} WMDP-Bio questions."
    )

    output_path = config.CANDIDATE_DATA_PATH

    write_questions(
        questions,
        output_path,
    )

    print(
        f"Candidate pool written to: {output_path}"
    )