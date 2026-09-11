"""Project-wide constants shared across CLI stages.

Single source of truth — do not redeclare these in any other module.
"""

SUBJECTS = [
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
]
CHANCE = 1 / 6  # 6-way condition decoding (complex/simple sentence, consonant strings,
# jabberwocky, word list, pseudoword list)
DEFAULT_TASK_LABEL = "RSVPLanguage"
DEFAULT_TR = 2.0
DEFAULT_SPACE_LABEL = "T1w"
STRATEGIES = ("fmriprep", "fmriprep+denoised", "denoised+fmriprep")
