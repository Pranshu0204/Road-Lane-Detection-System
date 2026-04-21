# Road Lane Detection System

Real-time road lane detection from video using OpenCV, with per-frame JSON stream logging, confidence scoring, and annotated video output.

---


### Bugs fixed

| Bug | Location | Fix |
|---|---|---|
| `average_slope_intercept()` had no `return` statement | line 50 | Added return of `(lines, confidence)` |
| `make_coordinates()` integer division error: `int(y1 - intercept)/slope` | line 38 | Corrected to `int((y1 - intercept) / slope)` |
| `combo_image` blended 1-channel gray with 3-channel line image | main loop | Both inputs to `addWeighted` are now 3-channel BGR |
| No end-of-video guard — crashed when `vid.read()` returned `None` | main loop | Added `if not ret or frame is None: break` |
| Lines drawn on cropped edge image instead of original frame | `display_lines` call | Canvas now matches original frame dimensions and uses BGR |

---

## Stream processing additions

Every frame now produces a structured record:

```json
{
  "frame": 42,
  "timestamp_s": 1.401,
  "status": "DETECTED",
  "confidence": 1.0,
  "raw_lines_found": 18,
  "fps": 28.4
}
```

On exit, all frame records are written to `lane_detection_log.json` alongside a summary:

```json
{
  "summary": {
    "total_frames": 847,
    "fully_detected": 701,
    "partially_detected": 98,
    "no_lanes": 48,
    "detection_rate": 0.828,
    "avg_confidence": 0.871,
    "duration_s": 28.4
  },
  "frames": [...]
}
```



---

## Confidence scoring

| Score | Status | Meaning |
|---|---|---|
| `1.0` | `DETECTED` | Both left and right lane lines found |
| `0.5` | `PARTIAL` | Only one lane line found |
| `0.0` | `NO_LANES` | No lane lines detected in this frame |

Confidence is computed from the number of valid averaged lane lines returned, not from raw Hough line count.

---

## Setup

```bash
pip install opencv-python numpy
```

No other dependencies required.

---

## Usage

```bash
# Default — looks for test_video.mp4 in the current directory
python3 RoadLaneDetection.py

# Custom video path via environment variable
VIDEO_PATH=/Users/pranshu/Downloads/test_video.mp4 python3 RoadLaneDetection.py
```

Press `q` to quit. On exit:
- `lane_detection_log.json` — full per-frame log + summary
- `output_annotated.mp4` — annotated video with lane overlays and HUD

To disable video output (faster, no disk write):
```python
SAVE_OUTPUT = False   # line 24 in RoadLaneDetection.py
```

---

## Pipeline architecture

```
Video file / camera stream
        │
        ▼
  vid.read() ──► frame (BGR)
        │
        ├──► process_image()         Grayscale → GaussianBlur → Canny edges
        │         │
        │         ▼
        │    region_of_interest()    Mask to road triangle ROI
        │         │
        │         ▼
        │    HoughLinesP()           Raw line segments
        │         │
        │         ▼
        │    average_slope_intercept()   Average into left + right lane lines
        │         │                     Returns (lines, confidence)
        │         ▼
        │    display_lines()         Draw onto blank 3-channel canvas
        │         │
        │         ▼
        │    addWeighted()           Blend with original frame
        │         │
        ├──► overlay_stats()         Burn frame/status/confidence/fps onto frame
        │
        ├──► cv2.imshow()            Live display
        ├──► VideoWriter.write()     Save annotated frame
        └──► frame_log.append()     Append JSON record
                  │
                  ▼ (on exit)
          lane_detection_log.json   Full stream log + summary
```

---

