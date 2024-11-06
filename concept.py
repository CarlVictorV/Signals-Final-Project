import imutils
import cv2
import numpy as np
import time
import os

# Create directory for storing motion frames if it doesn't exist
output_dir = './motion_frames'
os.makedirs(output_dir, exist_ok=True)

# Motion detection parameters
MIN_SIZE_FOR_MOVEMENT = 1000  # This size will not detect small motions like eye blinks
# MIN_SIZE_FOR_MOVEMENT = 200 # This is the minimum size for detecting motion

# Initialize webcam/video source
source = cv2.CAP_ANY  # Use any available camera
cap = cv2.VideoCapture(source)

# Initialize frame variables
reference_frame = None
motion_detected = False
start_time = time.time()
motion_start_time = None
initial_phase = True  # To track the phase (initial countdown vs. red light)

# Capture frames with a 5-second timer
while True:
    ret, frame = cap.read()
    if not ret:
        print("Capture error")
        break

    # Copy the original frame for drawing bounding boxes later
    original_frame = frame.copy()

    # Resize, grayscale, and blur the frame for better processing
    frame = imutils.resize(frame, width=750)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Countdown for capturing reference frame
    elapsed_time = time.time() - start_time
    if initial_phase:
        time_remaining = 5 - elapsed_time
        cv2.putText(original_frame, f"{int(time_remaining) + 1} seconds before Red Light", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Capture reference frame after countdown
        if time_remaining <= 0:
            reference_frame_s = original_frame.copy()  # Save reference in color
            reference_frame = gray
            print("Reference frame captured")
            motion_start_time = time.time()  # Start tracking for motion
            initial_phase = False  # Switch to motion detection phase
        else:
            cv2.imshow("frame", original_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue  # Skip further processing until Red Light phase begins

    # Motion detection phase
    elapsed_time = time.time() - motion_start_time
    time_remaining = 5 - elapsed_time
    cv2.putText(original_frame, f"Red Light - {int(time_remaining) + 1} seconds remaining",
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Compare current frame to reference
    if reference_frame is not None:
        frame_delta = cv2.absdiff(reference_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Detect contours
        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) > MIN_SIZE_FOR_MOVEMENT:
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(original_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                motion_detected = True

        # If motion is detected, save frames and stop
        if motion_detected:
            timestamp = int(time.time())
            cv2.imwrite(f"{output_dir}/{timestamp}_reference_frame.jpg", reference_frame_s)  # Save reference
            cv2.imwrite(f"{output_dir}/{timestamp}_motion_frame.jpg", original_frame)  # Save motion frame
            print("Motion detected! Saved reference and motion frames.")
            break

        # Display messages based on motion detection
        if motion_detected:
            cv2.putText(original_frame, "MOTION DETECTED", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        else:
            cv2.putText(original_frame, "NO MOTION", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Check if 5 seconds have passed without detecting motion
    if time_remaining < 0 and not motion_detected:
        print("No movement detected, safe.")
        cv2.putText(original_frame, "Safe! No Movement Detected", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("frame", original_frame)
        cv2.waitKey(2000)  # Show message for 2 seconds
        break

    # Display frame
    cv2.imshow("frame", original_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
