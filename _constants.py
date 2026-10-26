"""Project-wide constants shared across CLI stages.

Single source of truth — do not redeclare these in any other module.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Dataset:
    """Per-dataset defaults for the CLI stages.

    `block_size` is the number of consecutive same-condition trials that form
    one exchangeable unit for the single-subject test (`analysis.stats`): 1
    when trials are separate events, 4 for the stroke task's mini-blocks of
    four contractions of the same limb. `training_ps` are the numbers of
    held-out runs for the training-size analysis (p=1 is the main decode).
    """

    subjects: tuple[str, ...]
    task_label: str
    tr: float
    block_size: int
    training_ps: tuple[int, ...]


DATASETS = {
    # IBC RSVP language task: 6 runs x 60 trials, 6-way condition decoding
    # (complex/simple sentence, jabberwocky, word list, pseudoword list,
    # consonant strings).
    "ibc": Dataset(
        subjects=(
            "01",
            "02",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "11",
            "12",
            "13",
            "14",
            "15",
        ),
        task_label="RSVPLanguage",
        tr=2.0,
        block_size=1,
        training_ps=(1, 2, 3, 4),
    ),
    # 7T stroke patients, 4-limb motor task: 3 runs x 96 trials in mini-blocks
    # of 4 trials of the same limb, 4-way limb decoding.
    "motif": Dataset(
        subjects=("p01", "p02", "p03"),
        task_label="motif4limbs",
        tr=1.6,
        block_size=4,
        training_ps=(1, 2),
    ),
}

SUBJECTS = list(DATASETS["ibc"].subjects)
DEFAULT_TASK_LABEL = DATASETS["ibc"].task_label
DEFAULT_TR = DATASETS["ibc"].tr
DEFAULT_SPACE_LABEL = "T1w"
STRATEGIES = ("fmriprep", "fmriprep+denoised")

# Display names for figures/reports; strategies not listed fall back to their
# directory name.
STRATEGY_LABELS = {
    "fmriprep": "fMRIPrep",
    "fmriprep+denoised": "fMRIPrep + denoising",
    "ps7_ovmax_tfull_center": "fMRIPrep + LLR denoising",
}
# Both datasets' conditions (their names don't overlap). Ordered so that
# neighbouring rows of a confusion matrix are the conditions expected to be
# confused: language from most to least linguistic structure, hands then feet.
CONDITION_LABELS = {
    "ComplexSentence": "Complex sent.",
    "SimpleSentence": "Simple sent.",
    "Jabberwocky": "Jabberwocky",
    "WordList": "Word list",
    "PseudowordList": "Pseudoword list",
    "ConsonantStrings": "Consonant str.",
    "RightHand": "Right hand",
    "LeftHand": "Left hand",
    "RightFoot": "Right foot",
    "LeftFoot": "Left foot",
}
