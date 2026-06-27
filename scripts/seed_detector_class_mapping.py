"""Class-mapping rules for the seed detector.

Every source class name is mapped to exactly one of three target buckets
(drone / worker / enemy) by case-insensitive substring matching. The first
rule that matches wins. Anything that matches no rule is dropped with a
warning so it can be inspected manually.

Pollen was originally in this list but no source dataset actually exported
pollen annotations in COCO, so it is dropped from the target schema until
real pollen data is added. To add it back, drop ``POLLEN`` into ``TARGETS``
and ensure ``_RULES`` still covers it.

To change a mapping, edit ``_RULES``. Keep rules ordered most-specific first
because ``"worker"`` will also match ``"dead worker"``.
"""

from __future__ import annotations

DRONE = "drone"
WORKER = "worker"
ENEMY = "enemy"

TARGETS: tuple[str, ...] = (DRONE, WORKER, ENEMY)
TARGET_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(TARGETS)}

# (substring, bucket) — order matters, most specific first.
_RULES: tuple[tuple[str, str], ...] = (
    (DRONE, DRONE),
    ("hornet", ENEMY),
    ("yellowjacket", ENEMY),
    ("wasp", ENEMY),
    ("predator", ENEMY),
    ("intruder", ENEMY),
    ("foreign", ENEMY),
    (ENEMY, ENEMY),
    (WORKER, WORKER),
    ("forager", WORKER),
    ("guard", WORKER),
    ("fanning", WORKER),
    ("bee", WORKER),
)


def map_class(name: str) -> str | None:
    """Return the target bucket for ``name``, or ``None`` if it should be dropped."""
    lower = name.lower()
    for needle, bucket in _RULES:
        if needle in lower:
            return bucket
    return None


def describe_rules() -> list[tuple[str, str]]:
    """Return the rule table (substring -> bucket) for docs and audit output."""
    return list(_RULES)
