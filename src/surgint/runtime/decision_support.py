from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Optional, Union

from surgint.decision import (
    Decision,
    DecisionResolver,
    ItemContext,
    ItemContextOverride,
    Observation,
    SessionContext,
)
from surgint.runtime.inventory import (
    FinalInventory,
    FinalInventoryItem,
    Inventory,
    InventoryItem,
)

Labels = Union[Mapping[int, str], Sequence[str]]
Contexts = Union[ItemContext, Mapping[int, ItemContext]]


@dataclass(frozen=True)
class FinalInventoryDecision:
    """One finalized class estimate and its policy decision."""

    inventory_item: FinalInventoryItem
    decision: Decision

    @property
    def class_id(self) -> int:
        return self.inventory_item.class_id


@dataclass(frozen=True)
class FinalInventoryDecisions:
    """Immutable post-session inventory and decision result."""

    frames: int
    items: tuple[FinalInventoryDecision, ...]

    def by_class(self) -> dict[int, FinalInventoryDecision]:
        """Return finalized results keyed by perception class id."""

        return {item.class_id: item for item in self.items}


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

    def resolve_final_inventory(
        self,
        inventory: FinalInventory,
        context: Union[ItemContext, SessionContext],
        overrides: Optional[Mapping[int, ItemContextOverride]] = None,
    ) -> FinalInventoryDecisions:
        """Resolve one decision for each finalized class estimate."""

        if not isinstance(inventory, FinalInventory):
            raise TypeError("inventory must be a FinalInventory")
        contexts = _final_contexts(inventory, context, overrides)

        results = tuple(
            FinalInventoryDecision(
                inventory_item=item,
                decision=self._resolve_final_item(
                    item,
                    contexts[item.class_id],
                ),
            )
            for item in inventory.items
        )
        return FinalInventoryDecisions(frames=inventory.frames, items=results)

    def _resolve_final_item(
        self,
        item: FinalInventoryItem,
        context: ItemContext,
    ) -> Decision:
        label = self.labels.get(item.class_id)
        if label is None:
            raise ValueError(
                f"inventory class_id {item.class_id} has no perception label"
            )

        observation = Observation(
            label=label,
            confidence=item.score,
            track_id=item.representative_track_id,
            perception_version=self.perception_version,
        )
        return self.resolver.resolve(observation, context)


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


def _final_contexts(
    inventory: FinalInventory,
    context: Union[ItemContext, SessionContext],
    overrides: Optional[Mapping[int, ItemContextOverride]],
) -> dict[int, ItemContext]:
    class_ids = {item.class_id for item in inventory.items}

    if isinstance(context, ItemContext):
        if overrides is not None:
            raise ValueError("class overrides require a SessionContext")
        return {class_id: context for class_id in class_ids}

    if not isinstance(context, SessionContext):
        raise TypeError("context must be an ItemContext or SessionContext")

    if overrides is None:
        override_map = {}
    elif isinstance(overrides, Mapping):
        override_map = dict(overrides)
    else:
        raise TypeError("overrides must be a mapping or None")

    for class_id, override in override_map.items():
        if (
            isinstance(class_id, bool)
            or not isinstance(class_id, int)
            or class_id < 0
        ):
            raise ValueError("override class ids must be non-negative integers")
        if not isinstance(override, ItemContextOverride):
            raise TypeError(
                f"override for class_id {class_id} must be an "
                "ItemContextOverride"
            )

    unknown = sorted(set(override_map) - class_ids)
    if unknown:
        raise ValueError(f"overrides reference absent class ids {unknown}")

    return {
        class_id: context.for_item(override_map.get(class_id))
        for class_id in class_ids
    }
