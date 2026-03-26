# Backward-compatibility shim — use `from artifacts.artifact import ...` in new code.
from backend.artifacts.artifact import *  # noqa: F401, F403
from backend.artifacts.artifact import Artifact, TextArtifact, UserArtifact, TableArtifact, PlotArtifact, ErrorArtifact, ArtifactStore, artifact_factory


