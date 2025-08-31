import cv2
import json
import argparse
import numpy as np

# Predefined cell size (same for all cells)
CELL_W = 12
CELL_H = 12

def main():
    parser = argparse.ArgumentParser(description="Camera grid cell viewer with exposure control")
    parser.add_argument("--json", required=True, help="Path to JSON file with center points")
    parser.add_argument("--exposure", type=float, default=None, help="Exposure value to set on camera")
    args = parser.parse_args()

    # Load list of center positions
    with open(args.json, "r") as f:
        centers = json.load(f)  # [{"cx": x, "cy": y}, ...]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    if args.exposure is not None:
        cap.set(cv2.CAP_PROP_EXPOSURE, args.exposure)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        h, w = frame.shape[:2]
        stats = []

        for i, c in enumerate(centers):
            cx, cy = c["cx"], c["cy"]
            x1 = max(0, cx - CELL_W // 2)
            x2 = min(w - 1, cx + CELL_W // 2)
            y1 = max(0, cy - CELL_H // 2)
            y2 = min(h - 1, cy + CELL_H // 2)

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            med = np.median(roi.reshape(-1, 3), axis=0).astype(int)
            r, g, b = med[2], med[1], med[0]
            stats.append((i, r, g, b))

            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.circle(display, (cx, cy), 2, (0, 0, 255), -1)

        # Side panel for stats
        panel_w = 250
        panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
        y = 20
        for i, r, g, b in stats:
            line = f"Cell {i}: R={r} G={g} B={b}"
            cv2.putText(panel, line, (5, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)
            y += 20

        combined = np.hstack((display, panel))
        cv2.imshow("Camera Grid Cells", combined)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
