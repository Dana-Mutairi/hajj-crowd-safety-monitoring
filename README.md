# Smart Crowd Safety Monitoring for Hajj and Umrah

## Project Overview

This project presents an AI-powered crowd safety and flow-monitoring system designed for Hajj and Umrah environments.

The system processes surveillance video to detect and track people, count entry and exit movements, measure crowd density, identify the most crowded zone, generate movement heatmaps, and blur visible faces for privacy.

It also produces an annotated video and structured reports containing crowd counts, occupancy levels, zone information, and alert timestamps.

## Project Objective

The objective is to support crowd-safety monitoring by automatically:

- Detecting and tracking people.
- Counting people crossing an entry/exit line.
- Measuring the current occupancy of a monitored area.
- Classifying crowd density as low, medium, or high.
- Dividing the monitored area into three zones.
- Identifying the most crowded zone.
- Generating warnings when crowd density becomes high.
- Creating a movement heatmap.
- Blurring visible faces to protect privacy.
- Exporting results to CSV and JSON files.

## Main Features

### 1. Person Detection and Tracking

The pretrained YOLO26n model detects people in every video frame. ByteTrack assigns tracking IDs and follows detected people across consecutive frames.

### 2. Entry and Exit Counting

A virtual counting line is placed across the monitored passage. Ultralytics `ObjectCounter` counts tracked people crossing the line and classifies the movement as `IN` or `OUT`.

### 3. Crowd Density and Zone Monitoring

Ultralytics `RegionCounter` calculates the number of people currently inside the monitoring region.

The occupancy is classified into:

- Low density
- Medium density
- High density

The monitoring region is also divided into three zones. The system counts the people in each zone, assigns a safety status, and highlights the most crowded zone.

### 4. Movement Heatmap and Alerts

Ultralytics `Heatmap` accumulates person locations over time to show frequently used areas.

The system displays and records alerts when the crowd density exceeds the selected safety threshold.

### 5. Automatic Face Blurring

OpenCV YuNet detects visible faces. A Gaussian blur kernel is applied to the detected face regions to protect the privacy of people in the video.

## System Workflow

```text
Input Video
    ↓
Person Detection with YOLO26
    ↓
People Tracking with ByteTrack
    ↓
Entry and Exit Counting
    ↓
Occupancy and Crowd-Density Analysis
    ↓
Three-Zone Crowd Analysis
    ↓
Movement Heatmap
    ↓
Face Detection and Gaussian Blurring
    ↓
Crowd Alerts
    ↓
Processed Video and CSV/JSON Reports
```

## Tools and Libraries

- **Google Colab:** Cloud notebook environment used to develop and run the project.
- **Python:** Main programming language.
- **Ultralytics YOLO26:** Person detection and tracking.
- **ByteTrack:** Maintains person identities across video frames.
- **Ultralytics ObjectCounter:** Counts people crossing the virtual line.
- **Ultralytics RegionCounter:** Measures occupancy inside the monitored region.
- **Ultralytics Heatmap:** Generates the accumulated movement heatmap.
- **OpenCV:** Reads, processes, draws on, and writes video frames.
- **YuNet:** Detects visible faces.
- **NumPy:** Handles numerical calculations, coordinates, arrays, and zone boundaries.
- **Pandas:** Organizes and exports analytics data.
- **Matplotlib:** Produces analytics charts.
- **FFmpeg:** Converts the output video into the widely supported H.264 format.
- **JSON:** Stores crowd alerts and their timestamps.

## Crowd Monitoring Configuration

The project uses a main region of interest covering the visible pedestrian passage:

```python
crowd_region = [
    (250, 280),
    (1200, 280),
    (1200, 520),
    (250, 520)
]
```

The virtual entry and exit line is:

```python
counting_line = [
    (250, 430),
    (1200, 430)
]
```

The density thresholds were selected after analyzing the minimum, average, and maximum occupancy in the input video:

- Low: fewer than 21 people
- Medium: 21–27 people
- High: 28 or more people

The three-zone analysis uses the following initial thresholds:

- Safe: fewer than 7 people
- Warning: 7–9 people
- High: 10 or more people

These values are configurable and should be calibrated for the camera angle, monitored area, and operational safety requirements.

## How Zone Monitoring Works

The main region is divided into three equal vertical zones.

For every detected person, the system calculates the bottom-center point of the bounding box:

```python
person_x = int((x1 + x2) / 2)
person_y = int(y2)
```

This point approximately represents the person's position on the floor. Its coordinates determine which zone contains the person.

The zone containing the largest count is selected using:

```python
np.argmax(zone_counts)
```

and highlighted as the most crowded zone.

## How to Run

1. Open the notebook in Google Colab.
2. Select a GPU runtime if one is available:

```text
Runtime → Change runtime type → GPU
```

3. Place the input video inside the following Google Drive folder:

```text
MyDrive/Hajj_Crowd_Project/
```

4. Mount Google Drive.
5. Verify that `video_path` exactly matches the folder and video filename.
6. Install and import the required libraries.
7. Run the notebook cells in order.
8. Review the sample-frame detection before processing the complete video.
9. Run the integrated pipeline.
10. Open the generated video, CSV, JSON, and analytics files from Google Drive.

The notebook can run on a CPU, but complete video processing will take longer.

## Output Files

The project generates files such as:

- `hajj_crowd_monitoring_output.mp4` — integrated monitoring video.
- `hajj_crowd_summary.csv` — people counts and occupancy statistics.
- `hajj_crowd_alerts.json` — recorded alerts and timestamps.
- `hajj_crowd_analytics.png` — visual analytics chart.
- `hajj_final_results.csv` — final project results.
- `hajj_zone_monitoring.mp4` — three-zone monitoring video.
- `hajj_zone_summary.csv` — count and most-crowded-zone data.

## Example Results

For the tested video:

- Video resolution: 1280 × 576
- Frame rate: 30 FPS
- Duration: approximately 15 seconds
- Total processed frames: 449
- Sample-frame person detections: 35
- Minimum tested occupancy: 17
- Average tested occupancy: 24.16
- Maximum tested occupancy: 30
- Final live count: 17 IN and 4 OUT
- Recorded frame-level alerts: 9

Results depend on the input video, confidence threshold, camera perspective, occlusion, and visibility of the people.

## Exported Zone Data

The zone CSV includes fields such as:

```text
timestamp_seconds
zone_1_count
zone_2_count
zone_3_count
most_crowded_zone
total_inside_zones
```

The main analytics files include information such as:

- Timestamp
- Current occupancy
- Entry count
- Exit count
- Crowd-density level
- Alert status

## Technical Notes

- YOLO performs person detection; it does not directly determine crowd density.
- Crowd density is calculated from the number of detected people inside the monitored region.
- Tracking prevents the same person from being treated as a completely new person in every frame.
- Entry/exit counts and current occupancy represent different measurements.
- The heatmap represents accumulated movement, not only the people visible in the current frame.
- YuNet detects faces, while OpenCV applies the Gaussian blur.
- The alert rules are programmed using occupancy and zone thresholds.
- The final system integrates the separate solutions into one video-processing pipeline.

## Limitations

- Distant or heavily occluded people may not be detected.
- Tracking IDs can change when a person disappears behind another person.
- Small, covered, or unclear faces may not be detected by YuNet.
- Equal image zones do not necessarily represent equal physical areas because of camera perspective.
- Crowd thresholds are calibrated for the demonstration video and require adjustment for real deployment.
- `np.argmax()` selects the first zone if multiple zones share the same maximum count.
- Real deployment would require camera calibration, perspective correction, validation, and testing under different crowd conditions.

## Future Improvements

- Apply perspective correction for more accurate physical-zone measurements.
- Estimate crowd density per square metre.
- Add walking-speed and movement-direction estimation.
- Detect movement against the expected crowd flow.
- Combine feeds from multiple cameras.
- Build a live monitoring dashboard.
- Add real-time notification services.
- Use a specialized crowd-detection model for highly congested scenes.

## Training Program

This project was completed as part of a computer-vision training program covering:

- Image and video processing
- YOLO inference
- Object tracking
- Ultralytics Solutions
- Crowd analytics
- Model evaluation
- OpenCV and privacy protection
- End-to-end video processing
  
## Authors
Dana Almutairi
Hala Alhamzah
Joud Alazzaz

## SDAIA Academy

This project was completed within a training context connected to SDAIA Academy.

[SDAIA Academy on GitHub](https://github.com/SDAIAAcademy)

## Disclaimer

This project is an educational prototype. It should not be used as the sole basis for operational crowd-safety decisions without further validation, calibration, security review, and human supervision.
