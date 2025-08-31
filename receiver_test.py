import cv2
import json
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Camera cell viewer with exposure control")
    parser.add_argument("--json", required=True, help="Path to JSON file with cell definitions")
    parser.add_argument("--exposure", type=float, default=None, help="Exposure value to set on camera")
    args = parser.parse_args()

    # Load cell definitions
    with open(args.json, "r") as f:
        cells = json.load(f)  # [{"lt": [x1,y1], "rb": [x2,y2]}, ...]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    print("FPS reported:", cap.get(cv2.CAP_PROP_FPS))

    if args.exposure is not None:
        cap.set(cv2.CAP_PROP_EXPOSURE, args.exposure)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        h, w = frame.shape[:2]
        stats = []

        for i, cell in enumerate(cells):
            x1, y1 = cell["lt"]
            x2, y2 = cell["rb"]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            med = np.median(roi.reshape(-1, 3), axis=0).astype(int)
            r, g, b = med[2], med[1], med[0]  # OpenCV BGR→RGB
            stats.append((i, r, g, b))

            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Create side panel
        panel_w = 250
        panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
        y = 20
        for i, r, g, b in stats:
            line = f"Cell {i}: R={r} G={g} B={b}"
            cv2.putText(panel, line, (5, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)
            y += 20

        # Concatenate original and panel
        combined = np.hstack((display, panel))

        cv2.imshow("Camera Cells with Stats", combined)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
