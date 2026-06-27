"""Class-mapping rules for the seed detector.

Every source class name is mapped to exactly one of four target buckets
(drone / worker / pollen / enemy) by case-insensitive substring matching. The
first rule that matches wins. Anything that matches no rule is dropped with a
warning so it can be inspected manually.

To change a mapping, edit ``_RULES`` (or override via CLI in the merge
script). Keep rules ordered most-specific first because ``"worker"`` will also
match ``"dead worker"``.
"""

from __future__ import annotations

DRONE = "drone"
WORKER = "worker"
POLLEN = "pollen"
ENEMY = "enemy"

TARGETS: tuple[str, ...] = (DRONE, WORKER, POLLEN, ENEMY)
TARGET_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(TARGETS)}

# (substring, bucket) — order matters, most specific first.
_RULES: tuple[tuple[str, str], ...] = (
    (DRONE, DRONE),
    (POLLEN, POLLEN),
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
