import os
import cv2
from ultralytics import YOLO


# testi videon sijainti
video_path = "./test_video.mp4"

best_model = ""

for root, dirs, files in os.walk("runs"):
    for file in files:
        if file.endswith("best.pt"):
            best_model = os.path.join(root, file)

# best_model = "yolov8n.pt"

print(f"Using the best model: {best_model}")



model = YOLO(best_model)

cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:

        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()