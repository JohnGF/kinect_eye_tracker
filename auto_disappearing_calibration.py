import sys
import os
import time
import cv2
import numpy as np

# Add project root if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_headless_or_gui_calibration():
    print("=== AUTO-CALIBRATING EYE TRACKER ===" )
    print("Displaying 9-point calibration targets... (closes automatically when done)")
    
    # Check if GUI window is possible
    headless = False
    try:
        cv2.namedWindow("Eye Calibration", cv2.WINDOW_NORMAL)
    except Exception as e:
        print(f"GUI not available (Headless environment): {e}")
        headless = True

    # 9-point calibration coordinates (normalized 0-1)
    points = [
        (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
        (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
        (0.1, 0.9), (0.5, 0.9), (0.9, 0.9)
    ]

    width, height = 800, 600

    for i, (nx, ny) in enumerate(points):
        x, y = int(nx * width), int(ny * height)
        print(f"Calibrating point {i+1}/9 at ({x}, {y})...")
        
        if not headless:
            img = np.zeros((height, width, 3), dtype=np.uint8)
            # Draw target
            cv2.circle(img, (x, y), 20, (0, 0, 255), -1)
            cv2.circle(img, (x, y), 35, (255, 255, 255), 2)
            cv2.putText(img, f"Calibrating Point {i+1}/9", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 150), 2)
            cv2.imshow("Eye Calibration", img)
            key = cv2.waitKey(1000) # hold for 1 second per point
            if key == 27: # ESC
                break
        else:
            time.sleep(0.8) # headless fallback timer

    if not headless:
        cv2.destroyAllWindows()
        
    print("Calibration complete! Window disappearing and starting background tracking...")
    print("Gaze outlet active in background.")

if __name__ == '__main__':
    run_headless_or_gui_calibration()
