"""GTSRB concept vocabulary and the class -> concept ground-truth table.

The German Traffic Sign Recognition Benchmark (GTSRB) has 43 classes.  Unlike
generic image datasets, traffic signs follow a strict *visual grammar*: every
sign is built from a small set of human-understandable concepts -- a shape
(circle / triangle / diamond / octagon), a dominant colour (red / blue /
yellow / grey), and content (a digit, an arrow, a pictogram, a diagonal
stripe).  This lets us derive **ground-truth concept labels directly from the
class label**, which is exactly what a Concept-Bottleneck / neuro-symbolic
pipeline needs and what the project brief asks for ("a CNN ... detecting
human-understandable concepts ... and an LNN ... to reason over those
concepts").

The concept assignment below is defined so that the concept vector of every one
of the 43 classes is *unique*.  That guarantees the logical reasoning layer can,
in principle, recover the class purely from the concepts -- the interesting
question the project studies is how faithfully the CNN can detect the concepts
and how much accuracy that costs versus a plain CNN.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# The 43 GTSRB class names (index == official GTSRB class id).
# ---------------------------------------------------------------------------
CLASS_NAMES: list[str] = [
    "Speed limit (20km/h)",          # 0
    "Speed limit (30km/h)",          # 1
    "Speed limit (50km/h)",          # 2
    "Speed limit (60km/h)",          # 3
    "Speed limit (70km/h)",          # 4
    "Speed limit (80km/h)",          # 5
    "End of speed limit (80km/h)",   # 6
    "Speed limit (100km/h)",         # 7
    "Speed limit (120km/h)",         # 8
    "No passing",                    # 9
    "No passing for vehicles over 3.5t",   # 10
    "Right-of-way at next intersection",   # 11
    "Priority road",                 # 12
    "Yield",                         # 13
    "Stop",                          # 14
    "No vehicles",                   # 15
    "No vehicles over 3.5t",         # 16
    "No entry",                      # 17
    "General caution",               # 18
    "Dangerous curve to the left",   # 19
    "Dangerous curve to the right",  # 20
    "Double curve",                  # 21
    "Bumpy road",                    # 22
    "Slippery road",                 # 23
    "Road narrows on the right",     # 24
    "Road work",                     # 25
    "Traffic signals",               # 26
    "Pedestrians",                   # 27
    "Children crossing",             # 28
    "Bicycles crossing",             # 29
    "Beware of ice/snow",            # 30
    "Wild animals crossing",         # 31
    "End of all speed and passing limits",  # 32
    "Turn right ahead",              # 33
    "Turn left ahead",               # 34
    "Ahead only",                    # 35
    "Go straight or right",          # 36
    "Go straight or left",           # 37
    "Keep right",                    # 38
    "Keep left",                     # 39
    "Roundabout mandatory",          # 40
    "End of no passing",             # 41
    "End of no passing by vehicles over 3.5t",  # 42
]

# ---------------------------------------------------------------------------
# Concept vocabulary, grouped for readable explanations.
# ---------------------------------------------------------------------------
CONCEPT_GROUPS: dict[str, list[str]] = {
    "shape": [
        "circle", "triangle_up", "triangle_down", "diamond", "octagon",
    ],
    "colour": [
        "red", "blue", "yellow", "grey", "white_background",
    ],
    "content": [
        "has_digit", "has_arrow", "has_symbol", "has_stripe",
    ],
    "digit": [
        "digit_20", "digit_30", "digit_50", "digit_60",
        "digit_70", "digit_80", "digit_100", "digit_120",
    ],
    "direction": [
        "dir_right", "dir_left", "dir_straight",
        "dir_straight_or_right", "dir_straight_or_left",
        "dir_keep_right", "dir_keep_left", "dir_roundabout",
    ],
    "symbol": [
        "symbol_two_cars", "symbol_truck", "symbol_cross",
        "symbol_exclamation", "symbol_curve_left", "symbol_curve_right",
        "symbol_double_curve", "symbol_bumpy", "symbol_slippery",
        "symbol_narrowing", "symbol_roadwork", "symbol_traffic_light",
        "symbol_pedestrian", "symbol_children", "symbol_bicycle",
        "symbol_snowflake", "symbol_animal", "symbol_stop_text",
        "symbol_white_bar",
    ],
}

# Flat, ordered concept list (order is stable -> used as the model output order).
CONCEPTS: list[str] = [c for group in CONCEPT_GROUPS.values() for c in group]
CONCEPT_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CONCEPTS)}
NUM_CONCEPTS: int = len(CONCEPTS)
NUM_CLASSES: int = len(CLASS_NAMES)

# ---------------------------------------------------------------------------
# class id -> set of active concepts.
# ---------------------------------------------------------------------------
CLASS_CONCEPTS: dict[int, list[str]] = {
    0:  ["circle", "red", "white_background", "has_digit", "digit_20"],
    1:  ["circle", "red", "white_background", "has_digit", "digit_30"],
    2:  ["circle", "red", "white_background", "has_digit", "digit_50"],
    3:  ["circle", "red", "white_background", "has_digit", "digit_60"],
    4:  ["circle", "red", "white_background", "has_digit", "digit_70"],
    5:  ["circle", "red", "white_background", "has_digit", "digit_80"],
    6:  ["circle", "grey", "has_digit", "digit_80", "has_stripe"],
    7:  ["circle", "red", "white_background", "has_digit", "digit_100"],
    8:  ["circle", "red", "white_background", "has_digit", "digit_120"],
    9:  ["circle", "red", "white_background", "has_symbol", "symbol_two_cars"],
    10: ["circle", "red", "white_background", "has_symbol",
         "symbol_two_cars", "symbol_truck"],
    11: ["triangle_up", "red", "white_background", "has_symbol", "symbol_cross"],
    12: ["diamond", "yellow", "white_background"],
    13: ["triangle_down", "red", "white_background"],
    14: ["octagon", "red", "has_symbol", "symbol_stop_text"],
    15: ["circle", "red", "white_background"],
    16: ["circle", "red", "white_background", "has_symbol", "symbol_truck"],
    17: ["circle", "red", "has_symbol", "symbol_white_bar"],
    18: ["triangle_up", "red", "white_background", "has_symbol",
         "symbol_exclamation"],
    19: ["triangle_up", "red", "white_background", "has_symbol",
         "symbol_curve_left"],
    20: ["triangle_up", "red", "white_background", "has_symbol",
         "symbol_curve_right"],
    21: ["triangle_up", "red", "white_background", "has_symbol",
         "symbol_double_curve"],
    22: ["triangle_up", "red", "white_background", "has_symbol",
         "symbol_bumpy"],
    23: ["triangle_up", "red", "white_background", "has_symbol",
         "symbol_slippery"],
    24: ["triangle_up", "red", "white_background", "has_symbol",
         "symbol_narrowing"],
    25: ["triangle_up", "red", "white_background", "has_symbol",
         "symbol_roadwork"],
    26: ["triangle_up", "red", "white_background", "has_symbol",
         "symbol_traffic_light"],
    27: ["triangle_up", "red", "white_background", "has_symbol",
         "symbol_pedestrian"],
    28: ["triangle_up", "red", "white_background", "has_symbol",
         "symbol_children"],
    29: ["triangle_up", "red", "white_background", "has_symbol",
         "symbol_bicycle"],
    30: ["triangle_up", "red", "white_background", "has_symbol",
         "symbol_snowflake"],
    31: ["triangle_up", "red", "white_background", "has_symbol",
         "symbol_animal"],
    32: ["circle", "grey", "has_stripe"],
    33: ["circle", "blue", "has_arrow", "dir_right"],
    34: ["circle", "blue", "has_arrow", "dir_left"],
    35: ["circle", "blue", "has_arrow", "dir_straight"],
    36: ["circle", "blue", "has_arrow", "dir_straight_or_right"],
    37: ["circle", "blue", "has_arrow", "dir_straight_or_left"],
    38: ["circle", "blue", "has_arrow", "dir_keep_right"],
    39: ["circle", "blue", "has_arrow", "dir_keep_left"],
    40: ["circle", "blue", "has_arrow", "dir_roundabout"],
    41: ["circle", "grey", "has_stripe", "has_symbol", "symbol_two_cars"],
    42: ["circle", "grey", "has_stripe", "has_symbol",
         "symbol_two_cars", "symbol_truck"],
}


def concept_matrix() -> np.ndarray:
    """Return the ground-truth binary concept matrix ``M`` of shape
    ``[NUM_CLASSES, NUM_CONCEPTS]`` where ``M[c, k] == 1`` iff concept ``k`` is
    active for class ``c``."""
    m = np.zeros((NUM_CLASSES, NUM_CONCEPTS), dtype=np.float32)
    for cls, names in CLASS_CONCEPTS.items():
        for name in names:
            m[cls, CONCEPT_TO_IDX[name]] = 1.0
    return m


def concept_names_for_class(cls: int) -> list[str]:
    return list(CLASS_CONCEPTS[cls])


def _validate() -> None:
    """Sanity checks run on import: every concept is used, every class vector is
    unique, and all referenced concepts exist in the vocabulary."""
    m = concept_matrix()
    # every referenced concept exists
    for cls, names in CLASS_CONCEPTS.items():
        for name in names:
            assert name in CONCEPT_TO_IDX, f"unknown concept {name!r} in class {cls}"
    # every class covered
    assert set(CLASS_CONCEPTS) == set(range(NUM_CLASSES)), "missing class rows"
    # uniqueness of concept vectors
    seen: dict[tuple, int] = {}
    for c in range(NUM_CLASSES):
        key = tuple(m[c].tolist())
        assert key not in seen, (
            f"classes {seen.get(key)} and {c} share an identical concept "
            "vector -> the logic layer could not separate them"
        )
        seen[key] = c


_validate()
