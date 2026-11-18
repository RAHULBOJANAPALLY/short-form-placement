"""Placement decision layer — the part the spec cares most about.

The decision is made by an LLM through LangChain's `with_structured_output`,
bound to the `PlacementDecision` Pydantic schema. Because the schema fields are
`Literal[...]` over the allowed vocabulary, the model is constrained to valid
values — there is no free-form text we parse or regex.

BONUS - provider swap: set LLM_PROVIDER=anthropic|openai (no code change).
BONUS - caching: identical "scenes" (same occupancy signature) reuse one
        decision, so a 45s clip with a mostly-still host costs only a handful of
        API calls and the output is naturally stable window-to-window.
"""

from __future__ import annotations

import os
from typing import Dict, List

from langchain_core.prompts import ChatPromptTemplate

from .schema import PlacementDecision
from .windowing import WindowAgg

SYSTEM = """You are a layout planner for vertical (9:16) short-form video.
Given where the host's face and hands are on screen, you choose where overlays
go so they NEVER cover the face or hands.

The screen is divided into a 6-row x 4-column grid.
Rows: r1 = very top ... r6 = very bottom. Columns: c1 = left ... c4 = right.
You are told which cells the host currently occupies.

Pick exactly one value for each field from these meanings:

caption (a text strip):
  bottom_pill  -> rows r5-r6  (use when the bottom is clear)
  top_pill     -> rows r1-r2  (use when the top is clear)
  center_pill  -> rows r3-r4  (last resort, only if top and bottom are busy)

broll (a region filled with secondary footage):
  full           -> whole frame (only if the host is essentially absent)
  top40          -> top 40% (use when the host is in the lower half)
  bottom60       -> bottom 60% (use when the host is in the upper third)
  pip_top        -> small inset, top-center
  pip_corner_tr  -> small inset, top-right corner
  side_right_40  -> right 40% (use when the host is on the left)

icon (a small badge): top_left | top_right | bottom_left | bottom_right
  Put it in the emptiest corner.

Rules:
- The caption strip and the icon must land in cells the host does NOT occupy.
- The B-roll region should overlap the host as little as possible.
- Prefer the most stable, conventional choice; do not get creative."""

HUMAN = """Window {start}-{end} ms.
Host face box [x,y,w,h] normalized: {face}
Host hand boxes: {hands}
Occupied grid cells (6x4): {cells}

Choose caption, broll, icon and give one short reasoning sentence."""



def _make_llm():
    provider = os.getenv("LLM_PROVIDER", "google").lower()

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY is missing. Check your .env file."
            )

        model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        model = os.getenv(
            "ANTHROPIC_MODEL",
            "claude-3-5-sonnet-latest"
        )

        return ChatAnthropic(
            model=model,
            temperature=0,
            max_tokens=512
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        return ChatOpenAI(
            model=model,
            temperature=0
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER: {provider!r}. "
        "Use 'google', 'anthropic', or 'openai'."
    )



def _signature(w: WindowAgg) -> str:
    """Discretize a window into a 'scene' so identical scenes share a decision."""
    face = "none"
    if w.face_box is not None:
        # round to a coarse grid so tiny jitter doesn't bust the cache
        face = ",".join(f"{round(v, 1)}" for v in w.face_box)
    return f"f={face}|cells={'|'.join(w.occupied)}"


class PlacementPlanner:
    def __init__(self) -> None:
        llm = _make_llm()
        self.chain = ChatPromptTemplate.from_messages(
            [("system", SYSTEM), ("human", HUMAN)]
        ) | llm.with_structured_output(PlacementDecision)
        self._cache: Dict[str, PlacementDecision] = {}

    def decide(self, w: WindowAgg) -> PlacementDecision:
        sig = _signature(w)
        if sig in self._cache:
            return self._cache[sig]
        decision: PlacementDecision = self.chain.invoke(
            {
                "start": w.start_ms,
                "end": w.end_ms,
                "face": w.face_box if w.face_box is not None else "host not in frame",
                "hands": w.hand_boxes or "none",
                "cells": ", ".join(w.occupied) or "none",
            }
        )
        self._cache[sig] = decision
        return decision

    def plan(self, windows: List[WindowAgg]) -> List[PlacementDecision]:
        return [self.decide(w) for w in windows]
