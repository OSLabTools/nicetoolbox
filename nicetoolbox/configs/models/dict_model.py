from typing import Generic, Iterator, Mapping, TypeVar

from pydantic import RootModel

K = TypeVar("K")
V = TypeVar("V")


class DictModel(RootModel[dict[K, V]], Mapping[K, V], Generic[K, V]):
    """
    Pydantic RootModel with dictionary helper functions.

    Used for configs that don't have any fields, except key-value pairs
    of same type entities (e.g. dataset_properties).
    """

    def __getitem__(self, key: K) -> V:
        return self.root[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self.root)

    def __len__(self) -> int:
        return len(self.root)

    # Dictionary like helpers methods
    def keys(self):
        return self.root.keys()

    def items(self):
        return self.root.items()

    def values(self):
        return self.root.values()
