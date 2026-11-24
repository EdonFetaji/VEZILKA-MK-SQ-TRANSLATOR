"""Dataset loaders. Importing this package registers every loader by name.

To add a source: subclass ``DatasetLoader`` in a new module, decorate it with
``@register_loader("<name>")`` and import the module below.
"""

from src.datasets import (  # noqa: F401  (imported for registration)
    flores,
    gazette,
    media_literacy,
    muse,
    ntrex,
    opus,
    phrasebook,
    verbis,
)
from src.datasets.base import COLUMNS, DatasetLoader, RecordKind

__all__ = ["COLUMNS", "DatasetLoader", "RecordKind"]
