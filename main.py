
"""CLI entry point.

Usage:
    python main.py --input sample.mp4 --output out/

Pipeline:
  1. Probe video metadata
  2. Per-frame face/hand detection (cached)
  3. Aggregate frames into ~500 ms windows
  4. Ask the LLM for a placement per window
  5. Temporal smoothing
  6. Write zones.json and annotated.mp4
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
from dotenv import load_dotenv

from src.annotate import annotate
from src.detection import detect_video
from src.placement import PlacementPlanner
from src.schema import VideoMeta, Window, ZonesFile
from src.smoothing import smooth
from src.windowing import build_windows


# Load .env from the same directory as main.py
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)


def check_api_configuration() -> None:
    """Validate Google Gemini API configuration."""

    provider = os.getenv("LLM_PROVIDER", "google").lower()

    if provider == "google":
        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY is not loaded. "
                "Check your .env file."
            )

        print("Google Gemini API key loaded successfully.")

    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER: {provider}. "
            "This main.py configuration expects 'google'."
        )


def probe(path: str) -> VideoMeta:
    """Read video metadata."""

    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    duration_ms = int(frames / fps * 1000) if fps else 0

    return VideoMeta(
        path=os.path.basename(path),
        durationMs=duration_ms,
        width=width,
        height=height,
        fps=round(fps, 3),
    )


def main() -> None:

    # Validate API configuration
    check_api_configuration()

    # Argument parser
    parser = argparse.ArgumentParser(
        description="Placement-zone tool for talking-head video."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input video."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Directory where output files will be saved."
    )

    parser.add_argument(
        "--window-ms",
        type=int,
        default=500,
        help="Window length in milliseconds (300-700)."
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore the detection cache."
    )

    args = parser.parse_args()

    if args.window_ms <= 0:
        parser.error("--window-ms must be greater than zero.")

    input_path = Path(args.input).expanduser()
    output_dir = Path(args.output).expanduser()

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input video not found: {input_path}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Probe video
    meta = probe(str(input_path))

    print(
        f"Video: {meta.width}x{meta.height} "
        f"@ {meta.fps} FPS, {meta.durationMs} ms"
    )

    # Step 2: Detect faces and hands
    print("Starting face and hand detection...")

    detections = detect_video(
        str(input_path),
        use_cache=not args.no_cache
    )

    # Step 3: Build time windows
    aggs = build_windows(
        detections,
        meta.fps,
        meta.durationMs,
        window_ms=args.window_ms
    )

    print(f"{len(aggs)} windows of approximately {args.window_ms} ms")

    # Step 4: Ask Gemini for placement decisions
    print("Generating placement decisions using Google Gemini...")

    planner = PlacementPlanner()
    decisions = smooth(planner.plan(aggs))

    # Step 5: Build output windows
    windows = [
        Window(
            startMs=a.start_ms,
            endMs=a.end_ms,
            faceBoxNorm=a.face_box,
            handBoxesNorm=a.hand_boxes,
            occupiedCellsGrid6x4=a.occupied,
            suggestedPlacements=d,
            reasoning=d.reasoning,
        )
        for a, d in zip(aggs, decisions)
    ]

    # Step 6: Write zones.json
    zones = ZonesFile(
        video=meta,
        windows=windows
    )

    zones_path = output_dir / "zones.json"

    with open(zones_path, "w", encoding="utf-8") as file:
        json.dump(
            zones.model_dump(),
            file,
            indent=2
        )

    print(f"Wrote {zones_path}")

    # Step 7: Generate annotated video
    out_video = output_dir / "annotated.mp4"

    annotate(
        str(input_path),
        str(out_video),
        detections,
        windows,
        meta.fps
    )

    print(f"Wrote {out_video}")
    print("Video placement pipeline completed successfully.")


if __name__ == "__main__":
    main()
