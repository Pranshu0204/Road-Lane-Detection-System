import cv2
import numpy as np
import json
import time
import os

# ── Config ────────────────────────────────────────────────────────────────────
VIDEO_PATH   = os.environ.get("/Users/pranshu/Downloads/test_video.mp4", "test_video.mp4")
LOG_PATH     = "lane_detection_log.json"
SAVE_OUTPUT  = True          # set False to skip writing output video
OUTPUT_PATH  = "output_annotated.mp4"

# ── Image processing helpers ──────────────────────────────────────────────────

def process_image(image):
    """Convert frame to edge map."""
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


def region_of_interest(image):
    """Mask everything outside the road triangle."""
    height   = image.shape[0]
    width    = image.shape[1]
    polygons = np.array([[(200, height), (width - 200, height), (width // 2, int(height * 0.6))]])
    mask     = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(image, mask)


def make_coordinates(image, line_parameters):
    """Convert slope/intercept to pixel coordinates."""
    try:
        slope, intercept = line_parameters
        if abs(slope) < 1e-5:       # guard against near-zero slope
            slope = 0.001
    except (TypeError, ValueError):
        slope, intercept = 0.001, 0

    y1 = image.shape[0]
    y2 = int(y1 * 3 / 5)
    
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    """
    Average detects Hough lines into one left and one right lane line.
    Returns (left_line, right_line, confidence) where confidence is 0.0–1.0.
    
    """
    left_fit  = []
    right_fit = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            if x2 == x1:            # skip perfectly vertical lines
                continue
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope, intercept = parameters[0], parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

    left_line  = None
    right_line = None

    if left_fit:
        left_line  = make_coordinates(image, np.average(left_fit,  axis=0))
    if right_fit:
        right_line = make_coordinates(image, np.average(right_fit, axis=0))

    # Confidence: 1.0 = both lanes found, 0.5 = one lane, 0.0 = none
    detected = sum([left_line is not None, right_line is not None])
    confidence = round(detected / 2.0, 2)

    valid_lines = [l for l in [left_line, right_line] if l is not None]
    return np.array(valid_lines) if valid_lines else None, confidence


def display_lines(image, lines):
    """Draw lane lines onto a blank canvas matching image dimensions."""
    
    line_image = np.zeros((*image.shape[:2], 3), dtype=np.uint8)
    if lines is not None:
        for line in lines:
            try:
                x1, y1, x2, y2 = line.reshape(4)
                cv2.line(line_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 10)
            except Exception:
                continue
    return line_image


def overlay_stats(frame, frame_num, confidence, status, fps):
    """Draw frame metadata onto the display image."""
    color = (0, 255, 0) if status == "DETECTED" else (0, 165, 255) if status == "PARTIAL" else (0, 0, 255)
    cv2.putText(frame, f"Frame: {frame_num}",         (20, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"Status: {status}",           (20, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"Confidence: {confidence:.2f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}",             (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    return frame

# ── Stream processing loop ────────────────────────────────────────────────────

def run(video_path=VIDEO_PATH):
    vid = cv2.VideoCapture(video_path)

    if not vid.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return

    # Output video writer setup
    writer = None
    if SAVE_OUTPUT:
        width  = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_in = vid.get(cv2.CAP_PROP_FPS) or 30
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps_in, (width, height))

    # Per-frame log (stream processing observable output)
    frame_log   = []
    frame_count = 0
    t_start     = time.time()
    prev_time   = t_start

    print("[INFO] Processing video stream. Press 'q' to quit.")

    while True:
        ret, frame = vid.read()

        
        if not ret or frame is None:
            print("[INFO] End of video stream.")
            break

        frame_count += 1
        now      = time.time()
        fps      = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        timestamp = round(now - t_start, 3)

        # ── Lane detection pipeline ──────────────────────────────────────────
        edges         = process_image(frame)
        cropped_edges = region_of_interest(edges)
        lines         = cv2.HoughLinesP(
            cropped_edges, 2, np.pi / 180, 100,
            np.array([]), minLineLength=40, maxLineGap=5
        )

        
        averaged_lines, confidence = average_slope_intercept(frame, lines)

        # Detection status label
        if confidence == 1.0:
            status = "DETECTED"
        elif confidence == 0.5:
            status = "PARTIAL"
        else:
            status = "NO_LANES"

        # ── Compose display frame ────────────────────────────────────────────
        
        line_image = display_lines(frame, averaged_lines)
       
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1.0, 0)
        combo_image = overlay_stats(combo_image, frame_count, confidence, status, fps)

        
        raw_line_count = len(lines) if lines is not None else 0
        frame_record = {
            "frame":           frame_count,
            "timestamp_s":     timestamp,
            "status":          status,
            "confidence":      confidence,
            "raw_lines_found": raw_line_count,
            "fps":             round(fps, 2)
        }
        frame_log.append(frame_record)

        # ── Output ───────────────────────────────────────────────────────────
        if writer:
            writer.write(combo_image)

        cv2.imshow("Road Lane Detection", combo_image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Quit signal received.")
            break

    # ── Cleanup and summary ───────────────────────────────────────────────────
    vid.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # Write JSON log
    detected_frames = [f for f in frame_log if f["status"] == "DETECTED"]
    partial_frames  = [f for f in frame_log if f["status"] == "PARTIAL"]
    avg_conf        = round(float(np.mean([f["confidence"] for f in frame_log])), 4) if frame_log else 0.0

    summary = {
        "total_frames":       frame_count,
        "fully_detected":     len(detected_frames),
        "partially_detected": len(partial_frames),
        "no_lanes":           frame_count - len(detected_frames) - len(partial_frames),
        "detection_rate":     round(len(detected_frames) / max(frame_count, 1), 4),
        "avg_confidence":     avg_conf,
        "duration_s":         round(time.time() - t_start, 2)
    }

    output = {"summary": summary, "frames": frame_log}

    with open(LOG_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print("\n── Detection Summary ─────────────────────────────")
    for k, v in summary.items():
        print(f"  {k:<25} {v}")
    print(f"\n[INFO] Full frame log saved to: {LOG_PATH}")
    if SAVE_OUTPUT:
        print(f"[INFO] Annotated video saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
