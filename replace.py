import cv2
import os
import pdb

# Set up tracker.
OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
}

tracker_type = "boosting"
tracker_getter = OPENCV_OBJECT_TRACKERS[tracker_type]
tracker = None

video_path = os.path.join(os.path.dirname(__file__), "ap_run.mp4")
cap = cv2.VideoCapture(video_path)

new_face = cv2.imread("smiley.png", cv2.IMREAD_UNCHANGED)


def box_to_yxhw(box):
    return ((bbox[1] + bbox[3] / 2, bbox[0] + bbox[2] / 2),
            (bbox[3], bbox[2]))


def draw_centered(frame, img, y, x, h, w):
    x, y, h, w = tuple(map(int, (x, y, h, w)))
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    h, w, _ = img.shape
    w_down, w_up = w // 2, w - w // 2
    h_down, h_up = h // 2, h - h // 2
    cond = img[:, :, 3] > 0  # 'ignore' fully tranparent pixels
    frame[y - h_up: y + h_down, x - w_up: x + w_down][cond] = img[:, :, :3][cond]


while True:
    is_ok, frame = cap.read()
    if not is_ok:
        print("no read, exiting")
        break

    # Start timer
    timer = cv2.getTickCount()

    if tracker is not None:
        # Update tracker
        is_ok, bbox = tracker.update(frame)

        # Draw bounding box
        if is_ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            # breakpoint()
            (y, x), (h, w) = box_to_yxhw(bbox)
            draw_centered(frame, new_face, y, x, h, w)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display result
    cv2.imshow("Tracking", frame)

    key = cv2.waitKey(30) & 0xFF
    if key == ord("s"):
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        init_box = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
        # start OpenCV object tracker using the supplied bounding box
        # coordinates, then start the FPS throughput estimator as well
        tracker = tracker_getter()
        tracker.init(frame, init_box)
    elif key == ord("q"):
        break
