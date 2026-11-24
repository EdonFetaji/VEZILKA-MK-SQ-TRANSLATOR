"""Training tasks (instruction formats). Importing this package registers every task by name.

To add a task: subclass ``TrainTask`` in a new module, decorate it with
``@register_task("<name>")`` and import the module below.
"""

from src.train_tasks import (  # noqa: F401  (imported for registration)
    definition_translation,
    shuffled_definition,
    term_definition,
    term_pair,
    translation,
)
from src.train_tasks.base import DIRECTIONS, TrainTask

__all__ = ["DIRECTIONS", "TrainTask"]
