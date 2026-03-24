from ultralytics import YOLO
import cv2
import sys
import time

MODEL_PATH = "model/tichu_yolov8l/weights/best.pt"
CAM_INDEX = 0

# Common Logitech C270 resolutions:
# 640x480, 1280x720
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

WINDOW_NAME = "YOLO Webcam"

# Default slider values
DEFAULT_BRIGHTNESS = 128
DEFAULT_CONTRAST = 128
DEFAULT_SATURATION = 128
DEFAULT_GAIN = 0
DEFAULT_EXPOSURE_SLIDER = 50
DEFAULT_CONF_X100 = 25

# Button state
reset_all_requested = False
button_rect = (0, 0, 0, 0)  # x1, y1, x2, y2

model = YOLO(MODEL_PATH)

# Open webcam with DirectShow on Windows
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    sys.exit(1)

# Request resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# Read back actual resolution
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera opened at {actual_width}x{actual_height}")

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, actual_width, actual_height)


def nothing(x):
    pass


def slider_to_exposure(slider_value: int) -> float:
    return -float(slider_value) / 10.0


def apply_camera_settings(cap, brightness, contrast, saturation, gain, exposure_slider):
    cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
    cap.set(cv2.CAP_PROP_CONTRAST, contrast)
    cap.set(cv2.CAP_PROP_SATURATION, saturation)
    cap.set(cv2.CAP_PROP_GAIN, gain)

    exp_value = slider_to_exposure(exposure_slider)
    cap.set(cv2.CAP_PROP_EXPOSURE, exp_value)

    print(
        f"Set brightness={brightness}, contrast={contrast}, "
        f"saturation={saturation}, gain={gain}, exposure={exp_value:.1f}"
    )
    print(
        "Readback:",
        f"brightness={cap.get(cv2.CAP_PROP_BRIGHTNESS):.3f}",
        f"contrast={cap.get(cv2.CAP_PROP_CONTRAST):.3f}",
        f"saturation={cap.get(cv2.CAP_PROP_SATURATION):.3f}",
        f"gain={cap.get(cv2.CAP_PROP_GAIN):.3f}",
        f"exposure={cap.get(cv2.CAP_PROP_EXPOSURE):.3f}",
    )


def reset_all_settings(cap):
    print("Resetting all settings...")

    # Reset GUI sliders first
    cv2.setTrackbarPos("Brightness", WINDOW_NAME, DEFAULT_BRIGHTNESS)
    cv2.setTrackbarPos("Contrast",   WINDOW_NAME, DEFAULT_CONTRAST)
    cv2.setTrackbarPos("Saturation", WINDOW_NAME, DEFAULT_SATURATION)
    cv2.setTrackbarPos("Gain",       WINDOW_NAME, DEFAULT_GAIN)
    cv2.setTrackbarPos("Exposure",   WINDOW_NAME, DEFAULT_EXPOSURE_SLIDER)
    cv2.setTrackbarPos("Conf x100",  WINDOW_NAME, DEFAULT_CONF_X100)

    # Try to restore auto exposure briefly
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    time.sleep(0.2)

    # Then apply the default manual values again
    apply_camera_settings(
        cap,
        DEFAULT_BRIGHTNESS,
        DEFAULT_CONTRAST,
        DEFAULT_SATURATION,
        DEFAULT_GAIN,
        DEFAULT_EXPOSURE_SLIDER
    )

    print(f"Confidence reset to {DEFAULT_CONF_X100 / 100:.2f}")


def mouse_callback(event, x, y, flags, param):
    global reset_all_requested, button_rect

    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1, x2, y2 = button_rect
        if x1 <= x <= x2 and y1 <= y <= y2:
            reset_all_requested = True


cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

# Camera setting sliders
cv2.createTrackbar("Brightness", WINDOW_NAME, DEFAULT_BRIGHTNESS, 255, nothing)
cv2.createTrackbar("Contrast",   WINDOW_NAME, DEFAULT_CONTRAST, 255, nothing)
cv2.createTrackbar("Saturation", WINDOW_NAME, DEFAULT_SATURATION, 255, nothing)
cv2.createTrackbar("Gain",       WINDOW_NAME, DEFAULT_GAIN, 255, nothing)
cv2.createTrackbar("Exposure",   WINDOW_NAME, DEFAULT_EXPOSURE_SLIDER, 100, nothing)
cv2.createTrackbar("Conf x100",  WINDOW_NAME, DEFAULT_CONF_X100, 100, nothing)

last_vals = None

while True:
    if reset_all_requested:
        reset_all_settings(cap)
        reset_all_requested = False
        last_vals = None  # force refresh

    vals = (
        cv2.getTrackbarPos("Brightness", WINDOW_NAME),
        cv2.getTrackbarPos("Contrast",   WINDOW_NAME),
        cv2.getTrackbarPos("Saturation", WINDOW_NAME),
        cv2.getTrackbarPos("Gain",       WINDOW_NAME),
        cv2.getTrackbarPos("Exposure",   WINDOW_NAME),
        cv2.getTrackbarPos("Conf x100",  WINDOW_NAME),
    )

    if vals != last_vals:
        brightness, contrast, saturation, gain, exposure, conf_x100 = vals
        apply_camera_settings(cap, brightness, contrast, saturation, gain, exposure)
        print(f"conf={conf_x100 / 100:.2f}")
        last_vals = vals

    ret, frame = cap.read()
    if not ret or frame is None:
        print("ERROR: Could not read frame.")
        break

    conf = cv2.getTrackbarPos("Conf x100", WINDOW_NAME) / 100.0

    results = model.predict(
        source=frame,
        conf=conf,
        verbose=False
    )

    annotated_frame = results[0].plot()

    h, w = annotated_frame.shape[:2]
    cv2.putText(
        annotated_frame,
        f"{w}x{h}  conf={conf:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # Draw clickable reset-all button
    button_w = 220
    button_h = 40
    margin = 10
    x1 = w - button_w - margin
    y1 = margin
    x2 = w - margin
    y2 = margin + button_h
    button_rect = (x1, y1, x2, y2)

    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (50, 50, 50), -1)
    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.putText(
        annotated_frame,
        "Reset All Settings",
        (x1 + 10, y1 + 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        "Click button or press 'r' to reset all | q = quit",
        (10, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1
    )

    cv2.imshow(WINDOW_NAME, annotated_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        reset_all_requested = True

cap.release()
cv2.destroyAllWindows()