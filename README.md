# Short-form placement-zone tool

Watches a portrait (9:16) talking-head video and produces:

1. **`annotated.mp4`** — same video, same length, same audio, with per-frame
   overlays: red box on the face, yellow boxes on hands, a green caption pill,
   a blue B-roll rectangle, and a corner label with the timestamp + chosen
   placements.
2. **`zones.json`** — a sidecar describing each ~500 ms window: the normalized
   face/hand boxes, the occupied 6×4 grid cells, and an LLM-chosen placement
   template (`caption` / `broll` / `icon`) with a one-line reason.

```bash
python main.py --input sample.mp4 --output out/
```

That single command does everything: detection → windowing → LLM placement →
annotated video + JSON. No notebooks, no manual frame extraction.

---

## Setup (under 5 minutes)

Requires **Python 3.10 or 3.11** (MediaPipe wheels are most reliable there) and
**ffmpeg** on your `PATH`.

```bash
# 1. system dependency for muxing audio
sudo apt install ffmpeg        # macOS: brew install ffmpeg

# 2. python deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. API key for the placement layer
cp .env.example .env           # then edit .env and paste your key

# 4. run
python main.py --input sample.mp4 --output out/
```

`.env`:

```
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-3-5-sonnet-latest
```

---

## Library choices

| Layer | Choice | Why |
|---|---|---|
| Face + hand detection | **MediaPipe** (`face_detection` + `hands`) | Runs comfortably on CPU, no GPU, returns **normalized** boxes/landmarks directly — exactly the 0..1 coordinate space the spec wants. |
| Video I/O + drawing | **OpenCV** | Frame-accurate read/write and simple semi-transparent overlays via `addWeighted`. |
| Audio | **ffmpeg** (subprocess) | OpenCV drops audio, so the video is written silently then the *original* audio track is copied back, guaranteeing the audio matches the input. |
| Placement decision | **LangChain** `with_structured_output` + a **Pydantic** schema | The decision is a real structured-output call, not a parsed free-form prompt (see below). |
| LLM | **Anthropic** by default, **OpenAI** swappable via env var | — |

---

## How the LangChain layer works (the important bit)

The decision is bound to a Pydantic schema whose fields are `Literal` types over
the exact allowed vocabulary (`src/schema.py`):

```python
class PlacementDecision(BaseModel):
    caption: Literal["bottom_pill", "top_pill", "center_pill"]
    broll:   Literal["full","top40","bottom60","pip_top","pip_corner_tr","side_right_40"]
    icon:    Literal["top_left","top_right","bottom_left","bottom_right"]
    reasoning: str

chain = prompt | llm.with_structured_output(PlacementDecision)
```

Because the vocabulary lives in the schema, the model **cannot** return an
out-of-vocabulary value — there is no regex or string-matching fallback anywhere.
The LLM receives a compact, numeric summary of each window (face box, hand boxes,
and the occupied 6×4 grid cells) and reasons about which regions are free. There
are no `if/else` placement rules; the grid is just the *input* the model reasons
over.

---

## Design decisions (ambiguities I resolved myself)

- **Window length = 500 ms.** Within the suggested 300–700 ms range. A talking
  head doesn't move fast enough to justify sub-300 ms windows, which only add
  flicker and LLM calls; >700 ms lags real motion. 500 ms (~15 frames @30fps)
  aggregates enough frames to be jitter-free. Configurable via `--window-ms`.
- **Per-frame detection, window-level decisions.** The annotated video needs an
  accurate face/hand box on *every* frame, so detection runs per frame; the same
  results are reused to build windows — detection runs once.
- **Face box per window = median** of the frames where a face was found. If a
  face is missing in the majority of a window's frames, `faceBoxNorm` is omitted
  (the host briefly left frame) and the LLM is told the host is absent.
- **Hand boxes in the sidecar** come from the frame nearest the window midpoint
  (a clean representative). The **occupancy grid**, by contrast, is the *union*
  of the face and every hand across the whole window — deliberately conservative,
  so placement avoids anything the hands swept past.
- **Grid occupancy threshold:** a cell counts as occupied if a box covers >10% of
  it, which keeps grazing edges from marking a cell busy. This reproduces the
  spec's own example occupancy exactly.
- **Drawing geometry** for each named placement lives in `src/geometry.py` and is
  used *only* to render the overlay so the choice can be eyeballed — it never
  feeds back into the decision.

---

## Bonuses implemented

- **Detection caching** — results are hashed by file content + params and cached
  in `.cache/`; a rerun on the same video skips detection entirely
  (`--no-cache` to force).
- **Provider swap via env var** — `LLM_PROVIDER=anthropic|openai`, no code change.
- **Temporal smoothing** — a per-field majority filter snaps a lone disagreeing
  window to its neighbours when they agree, killing 1-window flicker
  (`src/smoothing.py`). The scene-signature cache in the placement layer also
  means identical scenes get identical decisions, which is stabilising and cuts
  API calls to a handful per clip.

---

## Project layout

```
main.py              CLI + pipeline orchestration
src/schema.py        Pydantic models + the Literal placement vocabulary
src/detection.py     per-frame MediaPipe face/hand detection (+ cache)
src/windowing.py     aggregate frames into contiguous windows
src/grid.py          6x4 occupancy grid
src/placement.py     LangChain structured-output decision (+ provider swap, cache)
src/smoothing.py     temporal flicker filter
src/geometry.py      named placement -> drawable rectangle
src/annotate.py      draw overlays + mux original audio
out/                 deliverables: annotated.mp4 + zones.json
```

---

## Writeup

**What I'd improve with more time.** I'd batch the unique-scene LLM calls with
`chain.batch()` instead of looping, and add a confidence/temporal weight so a
brief detection dropout doesn't flip a window. The hand representative in the
sidecar is a single midpoint frame; a proper clustered envelope across the window
would be more faithful when hands move a lot. I'd also add a small test video to
CI so the whole pipeline is exercised on every commit, and expose the drawing
geometry in `zones.json` so a downstream planner gets pixel rectangles, not just
names.

**What was hardest.** Two things. First, keeping the LLM strictly inside the
vocabulary while still letting it *reason* — solved by pushing the constraint
into `Literal` schema fields rather than the prompt, so structured output
guarantees validity. Second, deciding what a "window" summary should contain: too
much raw per-frame data drowns the model, too little and it can't avoid the hands.
A coarse 6×4 occupancy grid plus the raw boxes turned out to be the right level of
abstraction, and discretising that grid into a cache key gave temporal stability
almost for free.
