
from dataclasses import dataclass
from typing import Literal

from . import config


Condition = Literal[
    "honest",
    "sandbagging",
    "counterfactual",
]


@dataclass(frozen=True)
class PromptSet:
    """
    The three experimental conditions used by MIMESIS.

    The question and answer choices are shared across conditions.
    Only the condition-specific context changes.
    """

    honest_context: str
    sandbagging_context: str
    counterfactual_context: str



#SHARED TASK TEMPLATE


# Keep the task structure identical across experimental
# conditions. The question and answer choices are inserted
# into this template.
# The exact contextual wording is intentionally kept separate
# because it will be selected and frozen during calibration.

BASE_TASK_TEMPLATE = """{question}

{choices}

Answer:"""



#CONDITION CONTEXTS


# These are placeholders until the literature-grounded
# candidate prompts are selected during Phase 3 calibration.
#
# Do NOT run the main experiment with placeholder contexts.

HONEST_CONTEXT = ""

SANDBAGGING_CONTEXT = ""

COUNTERFACTUAL_CONTEXT = ""



# PROMPT SET

CURRENT_PROMPTS = PromptSet(
    honest_context=HONEST_CONTEXT,
    sandbagging_context=SANDBAGGING_CONTEXT,
    counterfactual_context=COUNTERFACTUAL_CONTEXT,
)


# CHOICE FORMATTING

def format_choices(choices: list[str]) -> str:
    """
    Format multiple-choice answers consistently.

    Example:
        A. choice one
        B. choice two
        C. choice three
        D. choice four
    """

    labels = ["A", "B", "C", "D"]

    if len(choices) != len(labels):
        raise ValueError(
            f"Expected {len(labels)} choices, got {len(choices)}."
        )

    return "\n".join(
        f"{label}. {choice}"
        for label, choice in zip(labels, choices)
    )



#PROMPT CONSTRUCTION


def build_prompt(
    question: str,
    choices: list[str],
    condition: Condition,
    prompt_set: PromptSet = CURRENT_PROMPTS,
) -> str:
    """
    Build a rendered prompt for one experimental condition.

    The question and answer choices are identical across
    conditions. Only the condition-specific context changes.
    """

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    choices_text = format_choices(choices)

    if condition == "honest":
        context = prompt_set.honest_context

    elif condition == "sandbagging":
        context = prompt_set.sandbagging_context

    elif condition == "counterfactual":
        context = prompt_set.counterfactual_context

    else:
        raise ValueError(
            f"Unknown condition: {condition}"
        )

    task = BASE_TASK_TEMPLATE.format(
        question=question.strip(),
        choices=choices_text,
    )

    if context.strip():
        return f"{context.strip()}\n\n{task}"

    return task


def build_honest_prompt(
    question: str,
    choices: list[str],
    prompt_set: PromptSet = CURRENT_PROMPTS,
) -> str:
    """Build the Honest-condition prompt."""

    return build_prompt(
        question,
        choices,
        condition="honest",
        prompt_set=prompt_set,
    )


def build_sandbagging_prompt(
    question: str,
    choices: list[str],
    prompt_set: PromptSet = CURRENT_PROMPTS,
) -> str:
    """Build the Sandbagging-condition prompt."""

    return build_prompt(
        question,
        choices,
        condition="sandbagging",
        prompt_set=prompt_set,
    )


def build_counterfactual_prompt(
    question: str,
    choices: list[str],
    prompt_set: PromptSet = CURRENT_PROMPTS,
) -> str:
    """Build the Counterfactual-condition prompt."""

    return build_prompt(
        question,
        choices,
        condition="counterfactual",
        prompt_set=prompt_set,
    )


# PROMPT CONSISTENCY CHECKS

def validate_shared_task(
    question: str,
    choices: list[str],
    prompt_set: PromptSet = CURRENT_PROMPTS,
) -> None:
    """
    Verify that all conditions contain the same question and
    answer choices.

    This does not require the contexts to be identical.
    """

    prompts = {
        condition: build_prompt(
            question,
            choices,
            condition=condition,
            prompt_set=prompt_set,
        )
        for condition in (
            "honest",
            "sandbagging",
            "counterfactual",
        )
    }

    task = BASE_TASK_TEMPLATE.format(
        question=question.strip(),
        choices=format_choices(choices),
    )

    for condition, prompt in prompts.items():
        if not prompt.endswith(task):
            raise ValueError(
                f"{condition} prompt does not preserve "
                "the shared task structure."
            )


def validate_prompts(
    prompt_set: PromptSet = CURRENT_PROMPTS,
) -> None:
    """
    Validate prompt configuration before use.
    """

    if not isinstance(prompt_set, PromptSet):
        raise TypeError(
            "prompt_set must be a PromptSet."
        )

    # The main experiment must never accidentally run with
    # missing condition-specific contexts.
    for name, context in (
        ("honest", prompt_set.honest_context),
        ("sandbagging", prompt_set.sandbagging_context),
        ("counterfactual", prompt_set.counterfactual_context),
    ):
        if not isinstance(context, str):
            raise TypeError(
                f"{name}_context must be a string."
            )

# SIMPLE TEST


if __name__ == "__main__":
    validate_prompts()

    test_question = "What is the correct answer?"
    test_choices = [
        "First option",
        "Second option",
        "Third option",
        "Fourth option",
    ]

    validate_shared_task(
        test_question,
        test_choices,
    )

    print("MIMESIS prompt module is valid.")