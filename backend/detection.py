
from ultralytics import YOLO
import cv2
import csv
from pathlib import Path

# Load a pretrained YOLO model
model = YOLO("yolo11n.pt")

# Video location
video_path = Path("../data/sample_videos/match.mp4")

# Output locations
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

csv_path = output_dir / "player_tracking.csv"
video_output = output_dir / "tracked_match.mp4"

# Start player tracking
results = model.track(
    source=video_path,
    tracker="bytetrack.yaml",
    persist=True,
    conf=0.30,
    classes=[0],
    stream=True
)

tracking_data = []
writer = None
frame_number = 0

for result in results:
    frame_number += 1

    boxes = result.boxes

    if boxes is not None and boxes.id is not None:
        coordinates = boxes.xyxy.cpu().numpy()
        player_ids = boxes.id.int().cpu().tolist()
        confidences = boxes.conf.cpu().tolist()

        for box, player_id, confidence in zip(
            coordinates, player_ids, confidences
        ):
            x1, y1, x2, y2 = box

            # Calculate player centre
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            tracking_data.append([
                frame_number,
                player_id,
                center_x,
                center_y,
                confidence
            ])

    # Draw tracking boxes and IDs
    annotated_frame = result.plot()

    if writer is None:
        height, width = annotated_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(video_output),
            fourcc,
            30,
            (width, height)
        )

    writer.write(annotated_frame)

# Export tracking data
with open(csv_path, "w", newline="") as file:
    csv_writer = csv.writer(file)

    csv_writer.writerow([
        "frame",
        "player_id",
        "x",
        "y",
        "confidence"
    ])

    csv_writer.writerows(tracking_data)

if writer is not None:
    writer.release()

print("Analysis complete!")
print(f"Tracking data saved to: {csv_path}")
print(f"Annotated video saved to: {video_output}")

