from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class UserInfo:
    id: str
    email: str
    full_name: str | None


@dataclass
class TenantInfo:
    id: str
    name: str


@dataclass
class UserContext:
    user: UserInfo
    tenant: TenantInfo
    role: str


@dataclass
class Settings:
    sla_target_sec: int
    human_minutes_per_resolved_call: float
    cost_per_minute: float
    currency: str
    retention_days: int
    mask_caller: bool
    recordings_enabled: bool
    recordings_require_manager: bool
    updated_at: str | datetime


def settings_to_dict(settings: Settings) -> dict[str, Any]:
    updated_at = (
        settings.updated_at.isoformat()
        if isinstance(settings.updated_at, datetime)
        else settings.updated_at
    )
    return {
        "slaTargetSec": settings.sla_target_sec,
        "humanMinutesPerResolvedCall": settings.human_minutes_per_resolved_call,
        "costPerMinute": settings.cost_per_minute,
        "currency": settings.currency,
        "retentionDays": settings.retention_days,
        "maskCaller": settings.mask_caller,
        "recordingsEnabled": settings.recordings_enabled,
        "recordingsRequireManager": settings.recordings_require_manager,
        "updatedAt": updated_at,
    }
