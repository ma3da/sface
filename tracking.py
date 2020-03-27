import cv2
import os

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

tracker_type = "mil"
tracker = OPENCV_OBJECT_TRACKERS[tracker_type]()
init_box = None

# Read video
video_path = os.path.join(os.path.dirname(__file__), "ap_run.mp4")
cap = cv2.VideoCapture(video_path)


while True:
    is_ok, frame = cap.read()
    if not is_ok:
        print("no read, exiting")
        break

    # Start timer
    timer = cv2.getTickCount()

    # Update tracker
    is_ok, bbox = tracker.update(frame)

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    if init_box is not None:
        # Draw bounding box
        if is_ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

    # # Display FPS on frame
    # cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75,  (50, 170, 50), 2)

    # Display result
    cv2.imshow("Tracking", frame)

    key = cv2.waitKey(30) & 0xFF
    if key == ord("s"):
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        init_box = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
        # start OpenCV object tracker using the supplied bounding box
        # coordinates, then start the FPS throughput estimator as well
        tracker.init(frame, init_box)
        # fps = FPS().start()
    elif key == ord("q"):
        break
