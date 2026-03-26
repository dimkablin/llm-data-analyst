from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import uuid


@dataclass
class ArtifactRecord:
    artifact_type: str
    data: object
    text: str | None = None
    role: str = "ai"
    meta: dict[str, object] = field(default_factory=dict)
    artifact_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


