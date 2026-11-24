"""Name → class registries for dataset loaders and training tasks.

Config files refer to loaders and tasks by name (``loader: gazette``, ``task: term_definition``).
Implementations register themselves with the decorators below; importing ``src.datasets`` and
``src.train_tasks`` fills the registries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from src.datasets.base import DatasetLoader
    from src.train_tasks.base import TrainTask

_LOADERS: dict[str, type[DatasetLoader]] = {}
_TASKS: dict[str, type[TrainTask]] = {}

L = TypeVar("L", bound="type[DatasetLoader]")
T = TypeVar("T", bound="type[TrainTask]")


def register_loader(name: str):
    def wrap(cls: L) -> L:
        if name in _LOADERS:
            raise ValueError(f"dataset loader {name!r} registered twice")
        cls.name = name
        _LOADERS[name] = cls
        return cls

    return wrap


def register_task(name: str):
    def wrap(cls: T) -> T:
        if name in _TASKS:
            raise ValueError(f"train task {name!r} registered twice")
        cls.name = name
        _TASKS[name] = cls
        return cls

    return wrap


def _ensure_imported() -> None:
    import src.datasets
    import src.train_tasks  # noqa: F401


def get_loader(name: str) -> type[DatasetLoader]:
    _ensure_imported()
    try:
        return _LOADERS[name]
    except KeyError:
        raise KeyError(f"unknown dataset loader {name!r}; known: {sorted(_LOADERS)}") from None


def get_task(name: str) -> type[TrainTask]:
    _ensure_imported()
    try:
        return _TASKS[name]
    except KeyError:
        raise KeyError(f"unknown train task {name!r}; known: {sorted(_TASKS)}") from None


def loader_names() -> list[str]:
    _ensure_imported()
    return sorted(_LOADERS)


def task_names() -> list[str]:
    _ensure_imported()
    return sorted(_TASKS)
