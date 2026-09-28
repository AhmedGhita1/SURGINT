from collections.abc import Mapping
from typing import Optional, Union

from surgint.decision import ItemContext, ItemContextOverride, SessionContext
from surgint.model import Detections
from surgint.runtime.decision_support import (
    FinalInventoryDecisions,
    InventoryDecisionSupport,
)
from surgint.runtime.inventory import Inventory


class DecisionSupportSession:
    """Enforce collection followed by one post-session decision pass."""

    def __init__(self, support: InventoryDecisionSupport):
        if not isinstance(support, InventoryDecisionSupport):
            raise TypeError("support must be an InventoryDecisionSupport")

        self._support = support
        self._inventory = Inventory()
        self._result: Optional[FinalInventoryDecisions] = None

    @property
    def is_finalized(self) -> bool:
        return self._result is not None

    @property
    def frame_count(self) -> int:
        return self._inventory.frame

    @property
    def result(self) -> FinalInventoryDecisions:
        """Return the final result; partial results are never exposed."""

        if self._result is None:
            raise RuntimeError("session has not been finalized")
        return self._result

    def update(self, detections: Detections) -> None:
        """Record one tracked frame while the session is open."""

        self._require_open()
        if not isinstance(detections, Detections):
            raise TypeError("detections must be Detections")
        self._inventory.update(detections)

    def finalize(
        self,
        context: Union[ItemContext, SessionContext],
        overrides: Optional[Mapping[int, ItemContextOverride]] = None,
    ) -> FinalInventoryDecisions:
        """Freeze inventory and resolve decisions exactly once."""

        self._require_open()
        finalized = self._inventory.finalize()
        result = self._support.resolve_final_inventory(
            finalized,
            context,
            overrides,
        )
        self._result = result
        return result

    def _require_open(self) -> None:
        if self.is_finalized:
            raise RuntimeError("session has already been finalized")
