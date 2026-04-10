"""
AutonomyController — permission gating for daemon actions.

Brain analog: The orbitofrontal cortex (OFC) — evaluates the expected value
and risk of prospective actions before the motor system commits to execution.
Patients with OFC damage act impulsively; a well-calibrated OFC suppresses
high-risk actions that lack sufficient evidence of reward. The AutonomyController
mirrors this by gating each proposed action against the configured autonomy
level, queuing uncertain actions for explicit user approval.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from mnemon.daemon.config import AutonomyLevel, DaemonConfig, RiskLevel

logger = logging.getLogger(__name__)


class ProposedAction(BaseModel):
    """An action the daemon wants to perform, pending permission check."""

    id: UUID = Field(default_factory=uuid4)
    description: str
    risk_level: RiskLevel
    source: str = Field(description="Module that proposed this action.")
    context: dict[str, Any] = Field(default_factory=dict)
    proposed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    approved: bool | None = None  # None = pending, True = approved, False = denied


class PermissionResult(BaseModel):
    """Outcome of an autonomy check."""

    allowed: bool
    needs_approval: bool = False
    reason: str = ""


# Mapping: autonomy level -> maximum risk level allowed without approval
_AUTO_ALLOW: dict[AutonomyLevel, set[RiskLevel]] = {
    AutonomyLevel.PASSIVE: {RiskLevel.LOW},
    AutonomyLevel.SUGGEST: {RiskLevel.LOW},
    AutonomyLevel.SEMI_AUTO: {RiskLevel.LOW, RiskLevel.MEDIUM},
    AutonomyLevel.AUTONOMOUS: {RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH},
}


class AutonomyController:
    """Permission gating for daemon actions.

    Each action is checked against the configured autonomy level. Actions
    that exceed the permission level are queued for user approval.
    """

    def __init__(self, config: DaemonConfig) -> None:
        self._level = config.autonomy_level
        self._pending: dict[UUID, ProposedAction] = {}
        logger.info("AutonomyController initialised — level=%s", self._level)

    @property
    def level(self) -> AutonomyLevel:
        return self._level

    @level.setter
    def level(self, value: AutonomyLevel) -> None:
        logger.info("Autonomy level changed: %s -> %s", self._level, value)
        self._level = value

    def check(self, action: ProposedAction) -> PermissionResult:
        """Check whether *action* is permitted under the current autonomy level.

        Returns ALLOWED for auto-permitted actions, NEEDS_APPROVAL for actions
        that require user consent, and DENIED for CRITICAL actions under any
        level below AUTONOMOUS.
        """
        allowed_risks = _AUTO_ALLOW.get(self._level, {RiskLevel.LOW})

        if action.risk_level in allowed_risks:
            logger.debug(
                "Action ALLOWED: %s (risk=%s, level=%s)",
                action.description[:80],
                action.risk_level,
                self._level,
            )
            return PermissionResult(allowed=True, reason="auto-permitted")

        # CRITICAL actions always need approval regardless of level
        if action.risk_level == RiskLevel.CRITICAL:
            self._pending[action.id] = action
            logger.info(
                "Action NEEDS_APPROVAL (critical): %s",
                action.description[:80],
            )
            return PermissionResult(
                allowed=False,
                needs_approval=True,
                reason="critical actions always require approval",
            )

        # Non-critical but exceeds autonomy level
        self._pending[action.id] = action
        logger.info(
            "Action NEEDS_APPROVAL: %s (risk=%s, level=%s)",
            action.description[:80],
            action.risk_level,
            self._level,
        )
        return PermissionResult(
            allowed=False,
            needs_approval=True,
            reason=f"risk level {action.risk_level} exceeds autonomy level {self._level}",
        )

    def approve(self, action_id: UUID) -> bool:
        """Approve a pending action. Returns False if not found."""
        action = self._pending.pop(action_id, None)
        if action is None:
            return False
        action.approved = True
        logger.info("Action APPROVED: %s", action.description[:80])
        return True

    def deny(self, action_id: UUID) -> bool:
        """Deny a pending action. Returns False if not found."""
        action = self._pending.pop(action_id, None)
        if action is None:
            return False
        action.approved = False
        logger.info("Action DENIED: %s", action.description[:80])
        return True

    def get_pending(self) -> list[ProposedAction]:
        """Return all actions awaiting user approval."""
        return list(self._pending.values())

    def clear_pending(self) -> int:
        """Clear all pending approvals. Returns count cleared."""
        count = len(self._pending)
        self._pending.clear()
        return count
