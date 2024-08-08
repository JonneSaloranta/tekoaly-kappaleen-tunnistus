import os
import cv2
from tqdm import tqdm
from ultralytics import YOLO
from time import sleep

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

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = int(cap.get(cv2.CAP_PROP_FPS))
fps = 30
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video properties: {frame_width}x{frame_height}, {fps} FPS, {total_frames} frames")

sleep(5)

# Define the codec and create a VideoWriter object to save the video
output_video_path = "./predicted_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Loop through the video frames with tqdm progress bar
with tqdm(total=total_frames, desc="Processing Video", unit="frame") as pbar:
    while cap.isOpened():

        # Read a frame from the videos
        success, frame = cap.read()

        if success:
            # Run YOLO model on the frame
            results = model(frame)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Write the annotated frame to the output video
            out.write(annotated_frame)

            # Update progress bar
            pbar.update(1)
        else:
            # Break the loop if the end of the video is reached
            break

# Release the video capture and writer objects
cap.release()
out.release()

print(f"Saved predicted video to: {output_video_path}")
