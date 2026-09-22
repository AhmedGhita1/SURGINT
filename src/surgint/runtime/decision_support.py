from collections.abc import Mapping, Sequence
from typing import Optional, Union

from surgint.decision import Decision, DecisionResolver, ItemContext, Observation
from surgint.runtime.inventory import Inventory, InventoryItem

Labels = Union[Mapping[int, str], Sequence[str]]
Contexts = Union[ItemContext, Mapping[int, ItemContext]]


class InventoryDecisionSupport:
    """Adapt tracked inventory items to the decision-support boundary."""

    def __init__(
        self,
        labels: Labels,
        perception_version: str,
        resolver: Optional[DecisionResolver] = None,
    ):
        self.labels = _label_map(labels)
        if not isinstance(perception_version, str) or not perception_version.strip():
            raise ValueError("perception_version must be a non-empty string")
        self.perception_version = perception_version
        self.resolver = resolver if resolver is not None else DecisionResolver()

    def resolve_item(
        self,
        item: InventoryItem,
        context: ItemContext,
    ) -> Decision:
        """Resolve one inventory item using its current class and best score."""

        if not isinstance(item, InventoryItem):
            raise TypeError("item must be an InventoryItem")
        if not isinstance(context, ItemContext):
            raise TypeError("context must be an ItemContext")

        label = self.labels.get(item.class_id)
        if label is None:
            raise ValueError(
                f"inventory class_id {item.class_id} has no perception label"
            )

        observation = Observation(
            label=label,
            confidence=item.score,
            track_id=item.track_id,
            perception_version=self.perception_version,
        )
        return self.resolver.resolve(observation, context)

    def resolve_inventory(
        self,
        inventory: Inventory,
        contexts: Contexts,
    ) -> dict[int, Decision]:
        """Resolve every item with shared or track-specific context."""

        if not isinstance(inventory, Inventory):
            raise TypeError("inventory must be an Inventory")

        if isinstance(contexts, ItemContext):
            return {
                track_id: self.resolve_item(item, contexts)
                for track_id, item in inventory.items.items()
            }

        if not isinstance(contexts, Mapping):
            raise TypeError("contexts must be an ItemContext or a mapping")

        missing = sorted(set(inventory.items) - set(contexts))
        if missing:
            raise ValueError(f"missing item context for track ids {missing}")

        decisions = {}
        for track_id, item in inventory.items.items():
            context = contexts[track_id]
            if not isinstance(context, ItemContext):
                raise TypeError(
                    f"context for track_id {track_id} must be an ItemContext"
                )
            decisions[track_id] = self.resolve_item(item, context)
        return decisions


def _label_map(labels: Labels) -> dict[int, str]:
    if isinstance(labels, Mapping):
        mapping = dict(labels)
    elif isinstance(labels, Sequence) and not isinstance(labels, (str, bytes)):
        mapping = dict(enumerate(labels))
    else:
        raise TypeError("labels must be a mapping or sequence")

    if not mapping:
        raise ValueError("labels must not be empty")

    for class_id, label in mapping.items():
        if (
            isinstance(class_id, bool)
            or not isinstance(class_id, int)
            or class_id < 0
        ):
            raise ValueError("label-map class ids must be non-negative integers")
        if not isinstance(label, str) or not label.strip():
            raise ValueError("perception labels must be non-empty strings")

    values = list(mapping.values())
    duplicates = sorted({label for label in values if values.count(label) > 1})
    if duplicates:
        raise ValueError(f"perception labels must be unique; duplicates={duplicates}")
    return mapping
