import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import supervision as sv
import os
import datetime
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configuration
DEFAULT_SOURCE_VIDEO_PATH = "input.mp4"
DEFAULT_TARGET_VIDEO_PATH = "output_video4.mp4"
DEFAULT_CONFIDENCE_THRESHOLD = 0.3
DEFAULT_IOU_THRESHOLD = 0.7
DEFAULT_SPEED_THRESHOLD = 90  # Speed threshold in km/h
PHOTO_DIR = "speed_violation_photos3"
LOG_FILE = "speed_violations.log"
STATS_FILE = "speed_stats.png"

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Vehicle Speed Detection")
parser.add_argument("--source", default=DEFAULT_SOURCE_VIDEO_PATH, help="Path to source video")
parser.add_argument("--output", default=DEFAULT_TARGET_VIDEO_PATH, help="Path to output video")
parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD, help="Confidence threshold")
parser.add_argument("--iou", type=float, default=DEFAULT_IOU_THRESHOLD, help="IOU threshold")
parser.add_argument("--speed_limit", type=int, default=DEFAULT_SPEED_THRESHOLD, help="Speed limit in km/h")
args = parser.parse_args()

# Make sure the directory to save photos exists
os.makedirs(PHOTO_DIR, exist_ok=True)

# Perspective transform source and target points
SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])
TARGET_WIDTH = 25
TARGET_HEIGHT = 250
TARGET = np.array([
    [0, 0],
    [TARGET_WIDTH - 1, 0],
    [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
    [0, TARGET_HEIGHT - 1],
])

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

def log_speed_violation(tracker_id, speed):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"{timestamp} - Vehicle ID: {tracker_id}, Speed: {speed:.2f} km/h\n")

def capture_vehicle_photo(frame, detection, tracker_id, speed):
    x1, y1, x2, y2 = [int(coord) for coord in detection]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    vehicle_image = frame[y1:y2, x1:x2]
    filename = f"{PHOTO_DIR}/car_{tracker_id}_speed_{int(speed)}.jpg"
    cv2.imwrite(filename, vehicle_image)

def plot_speed_distribution(speeds):
    plt.figure(figsize=(10, 6))
    plt.hist(speeds, bins=20, edgecolor='black')
    plt.title('Speed Distribution of Vehicles')
    plt.xlabel('Speed (km/h)')
    plt.ylabel('Number of Vehicles')
    plt.axvline(args.speed_limit, color='r', linestyle='dashed', linewidth=2, label='Speed Limit')
    plt.legend()
    plt.savefig(STATS_FILE)
    plt.close()

def main():
    video_info = sv.VideoInfo.from_video_path(video_path=args.source)
    model = YOLO("yolov8s.pt")

    byte_track = sv.ByteTrack(
        frame_rate=video_info.fps, track_thresh=args.confidence
    )

    thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh) * 1.5
    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER,
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=video_info.fps * 2,
        position=sv.Position.BOTTOM_CENTER,
    )

    frame_generator = sv.get_video_frames_generator(source_path=args.source)

    polygon_zone = sv.PolygonZone(polygon=SOURCE)
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
    speed_status = {}  # Track whether a vehicle has been counted
    count_above_speed = 0
    count_below_speed = 0
    total_speed = 0
    vehicle_count = 0
    all_speeds = []

    progress_bar = tqdm(total=video_info.total_frames, desc="Processing frames")

    with sv.VideoSink(args.output, video_info) as sink:
        for frame_num, frame in enumerate(frame_generator):
            result = model(frame)[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = detections[detections.confidence > args.confidence]
            detections = detections[polygon_zone.trigger(detections)]
            detections = detections.with_nms(threshold=args.iou)
            detections = byte_track.update_with_detections(detections=detections)

            points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            points = view_transformer.transform_points(points=points).astype(int)

            for tracker_id, [_, y] in zip(detections.tracker_id, points):
                coordinates[tracker_id].append(y)

            labels = []
            for i, (tracker_id, detection) in enumerate(zip(detections.tracker_id, detections)):
                if len(coordinates[tracker_id]) < video_info.fps / 2:
                    labels.append(f"#{tracker_id}")
                else:
                    coordinate_start = coordinates[tracker_id][-1]
                    coordinate_end = coordinates[tracker_id][0]
                    distance = abs(coordinate_start - coordinate_end)
                    time = len(coordinates[tracker_id]) / video_info.fps
                    speed = distance / time * 3.6

                    if tracker_id not in speed_status:
                        if speed > args.speed_limit:
                            count_above_speed += 1
                            color = (0, 0, 255)  # Red for speeding cars
                            capture_vehicle_photo(frame, detections.xyxy[i], tracker_id, speed)
                            log_speed_violation(tracker_id, speed)
                            speed_status[tracker_id] = "above"
                        else:
                            count_below_speed += 1
                            color = (50, 255, 50)  # Green for cars within the speed limit
                            speed_status[tracker_id] = "below"
                        
                        total_speed += speed
                        vehicle_count += 1
                        all_speeds.append(speed)
                    else:
                        color = (255, 50, 50) if speed_status[tracker_id] == "above" else (50, 255, 50)

                    labels.append(f"#{tracker_id} {int(speed)} km/h")
                    box_annotator.annotate_color = color

            annotated_frame = frame.copy()
            annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

            # Semi-transparent overlay for stats
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (0, 0), (400, 200), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, annotated_frame, 0.5, 0, annotated_frame)

            # Display the counts and average speed
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2  # Larger font scale
            font_thickness = 3  # Thicker font
            cv2.putText(annotated_frame, f"Speeding: {count_above_speed}", (20, 40),
                        font, font_scale, (255, 50, 50), font_thickness)
            cv2.putText(annotated_frame, f"Normal: {count_below_speed}", (20, 80),
                        font, font_scale, (50, 255, 50), font_thickness)
            
            avg_speed = total_speed / vehicle_count if vehicle_count > 0 else 0
            cv2.putText(annotated_frame, f"Avg Speed: {avg_speed:.2f} km/h", (20, 120),
                        font, font_scale, (200, 200, 255), font_thickness)

            # Progress bar
            cv2.rectangle(annotated_frame, (20, 160), (int(20 + (frame_num / video_info.total_frames) * 360), 180),
                          (0, 255, 255), -1)

            sink.write_frame(annotated_frame)
            progress_bar.update(1)

    progress_bar.close()
    plot_speed_distribution(all_speeds)

if __name__ == "__main__":
    main()
