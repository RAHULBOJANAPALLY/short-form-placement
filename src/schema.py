"""Pydantic models.

`PlacementDecision` is the schema the LLM is forced to fill via
LangChain's `with_structured_output`. The `Literal` types are what make the
vocabulary impossible to violate — the model literally cannot return a value
outside the allowed lists, so we never have to validate/regex the text.
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

# ---- Allowed vocabulary (kept as type aliases so they're used everywhere) ----

CaptionZone = Literal["bottom_pill", "top_pill", "center_pill"]
BrollZone = Literal[
    "full", "top40", "bottom60", "pip_top", "pip_corner_tr", "side_right_40"
]
IconZone = Literal["top_left", "top_right", "bottom_left", "bottom_right"]


class PlacementDecision(BaseModel):
    """What the LLM returns for a single time window."""

    caption: CaptionZone = Field(
        description="Where to put the caption pill so it does not cover the face/hands."
    )
    broll: BrollZone = Field(
        description="Which region to fill with B-roll so it does not cover the face/hands."
    )
    icon: IconZone = Field(
        description="Which corner to place a small icon, away from the subject."
    )
    reasoning: str = Field(
        description="One short sentence explaining the choice in terms of where the host is."
    )


# ---- Internal data carried through the pipeline (not LLM-facing) ----

# A normalized box is [x, y, w, h] with top-left origin, all in 0..1.
Box = List[float]


@dataclass
class FrameDetection:
    """Detections for a single video frame (plain data, no heavy deps)."""

    frame_idx: int
    face_box: Optional[Box] = None          # [x,y,w,h] normalized, or None
    hand_boxes: List[Box] = field(default_factory=list)


class Window(BaseModel):
    """One contiguous time window of the output sidecar."""

    startMs: int
    endMs: int
    faceBoxNorm: Optional[Box] = None
    handBoxesNorm: List[Box] = Field(default_factory=list)
    occupiedCellsGrid6x4: List[str] = Field(default_factory=list)
    suggestedPlacements: PlacementDecision
    reasoning: str


class VideoMeta(BaseModel):
    path: str
    durationMs: int
    width: int
    height: int
    fps: float


class ZonesFile(BaseModel):
    video: VideoMeta
    windows: List[Window]
