import cv2
import numpy as np

# ---------- Settings you can change ----------
CAMERA_INDEX = 0     # 0 = default webcam. If you see the wrong camera, try 1.
MIN_AREA = 500       # smallest red blob (in pixels) that counts as an "object"

# Red is special in HSV colours: it sits at BOTH ends of the hue scale,
# so we need two ranges and combine them.
LOWER_RED_1 = np.array([0, 120, 70])
UPPER_RED_1 = np.array([10, 255, 255])
LOWER_RED_2 = np.array([170, 120, 70])
UPPER_RED_2 = np.array([180, 255, 255])


def get_red_mask(frame):
    """Return a black-and-white picture: white = red pixels, black = everything else."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1)
    mask2 = cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
    mask = cv2.bitwise_or(mask1, mask2)
    # Remove tiny specks of noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return mask


def zone_of(x, y, width, height):
    """Which zone (0-3) is the point (x, y) in?   Zone 1 = top-left, 2 = top-right,
    3 = bottom-left, 4 = bottom-right."""
    column = 0 if x < width // 2 else 1
    row = 0 if y < height // 2 else 1
    return row * 2 + column


def detect_zones(frame):
    """Find red objects and return which of the 4 zones are occupied."""
    mask = get_red_mask(frame)
    height, width = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    occupied = [False, False, False, False]
    centers = []
    for contour in contours:
        if cv2.contourArea(contour) < MIN_AREA:
            continue  # too small, probably noise
        m = cv2.moments(contour)
        cx = int(m['m10'] / m['m00'])   # centre of the object
        cy = int(m['m01'] / m['m00'])
        occupied[zone_of(cx, cy, width, height)] = True
        centers.append((cx, cy))
    return occupied, centers, mask


def draw_overlay(frame, occupied, centers, actual_level):
    """Draw the zone lines, labels and the Actual level on the picture."""
    height, width = frame.shape[:2]
    cv2.line(frame, (width // 2, 0), (width // 2, height), (255, 255, 255), 2)
    cv2.line(frame, (0, height // 2), (width, height // 2), (255, 255, 255), 2)

    corners = [(10, 30), (width // 2 + 10, 30),
               (10, height // 2 + 30), (width // 2 + 10, height // 2 + 30)]
    for i in range(4):
        text = f"Zone {i + 1}: {'OCCUPIED' if occupied[i] else 'empty'}"
        colour = (0, 255, 0) if occupied[i] else (200, 200, 200)
        cv2.putText(frame, text, corners[i], cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

    for (cx, cy) in centers:
        cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)

    cv2.putText(frame, f"ACTUAL LEVEL = {actual_level}", (10, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Could not open the webcam. Try changing CAMERA_INDEX to 1.")
        return

    print("Camera running. Press 'q' in the camera window to quit.")
    last_level = None
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Could not read from the webcam.")
            break

        frame = cv2.flip(frame, 1)   # mirror view, so left/right feel natural
        occupied, centers, mask = detect_zones(frame)
        actual_level = sum(occupied)   # number of occupied zones = 0 to 4

        if actual_level != last_level:   # only print when it changes
            print(f"Occupied zones: {[i + 1 for i in range(4) if occupied[i]]}"
                  f"  ->  Actual level = {actual_level}")
            last_level = actual_level

        draw_overlay(frame, occupied, centers, actual_level)
        cv2.imshow("Smart Classroom Camera", frame)
        cv2.imshow("Red mask (what the camera thinks is red)", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
