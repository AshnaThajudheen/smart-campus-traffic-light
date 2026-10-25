# 🚦 Smart Campus Traffic Light System

### AI-Based Vehicle Detection and Adaptive Traffic Signal Control

An AI-powered traffic monitoring system designed to detect vehicles, estimate congestion, and recommend adaptive traffic signal timings. The project aims to improve traffic flow and safety at campus entrances using computer vision and intelligent decision-making.

🌐 **Live Demo:** [Smart Campus Traffic Light System](https://smart-campus-traffic-light.streamlit.app/)

---

## 📌 Project Overview

Traditional traffic lights often operate with fixed timings, regardless of the number of vehicles waiting. This can lead to unnecessary delays and congestion, particularly at busy campus entrances.

The Smart Campus Traffic Light System uses computer vision to detect vehicles in traffic videos, estimate traffic density, and recommend green-light durations based on observed congestion.

The initial prototype uses a rule-based adaptive controller, with reinforcement learning planned as a future enhancement.

## 🎯 Objectives

- Detect vehicles in traffic videos using deep learning.
- Identify cars, motorcycles, buses, and trucks.
- Estimate traffic congestion from vehicle detections.
- Recommend adaptive traffic signal timings.
- Visualize traffic analysis through an interactive web application.
- Explore reinforcement learning for intelligent traffic management.

## ✨ Features

- **Vehicle Detection:** Detects vehicles using a pretrained YOLO model.
- **Video Analysis:** Processes uploaded traffic videos.
- **Congestion Estimation:** Classifies traffic as low, medium, or high.
- **Adaptive Signal Recommendation:** Suggests green-light durations based on detected traffic.
- **Interactive Dashboard:** Displays vehicle counts and traffic analysis.
- **Data Export:** Allows users to download detection results as a CSV file.
- **Web Deployment:** Accessible through Streamlit Community Cloud.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| YOLOv8 | Vehicle detection |
| OpenCV | Video processing and annotation |
| Streamlit | Interactive web application |
| Pandas | Traffic data analysis |
| Matplotlib | Data visualization |
| NumPy | Numerical operations |
| GitHub | Version control and source code hosting |
| Streamlit Community Cloud | Web application deployment |

## ⚙️ System Workflow

1. Upload a traffic video.
2. Process video frames using the YOLO object detection model.
3. Detect and count vehicles in each frame.
4. Estimate traffic levels from the observed detections.
5. Recommend a green-light duration using adaptive rules.
6. Display the analysis and allow the results to be downloaded.

## 🏗️ System Architecture

```text
Traffic Video
     |
     v
Video Processing (OpenCV)
     |
     v
Vehicle Detection (YOLOv8)
     |
     v
Vehicle Count and Traffic Analysis
     |
     v
Congestion Estimation
     |
     v
Rule-Based Adaptive Signal Controller
     |
     v
Recommended Signal Timing
     |
     v
Streamlit Dashboard and CSV Report
```

## 🚀 How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/smart-campus-traffic-light.git
cd smart-campus-traffic-light
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal.

## 📊 Current Implementation

The current prototype supports:

- Video upload and processing.
- YOLO-based vehicle detection.
- Frame-level vehicle counts.
- Basic congestion classification.
- Rule-based green-light recommendations.
- Traffic visualization and CSV export.

**Note:** Frame-level detections are not unique vehicle counts. The same vehicle can be detected across multiple frames. The current controller analyzes one camera view and does not yet implement a complete multi-lane signal cycle.

## 🔮 Future Enhancements

- Vehicle tracking to count unique vehicles.
- Multi-lane and multi-approach congestion estimation.
- Waiting-time-aware adaptive signal control.
- Reinforcement learning using a traffic simulator such as SUMO.
- Comparison of fixed-time, rule-based, and reinforcement-learning controllers.
- Performance evaluation using average waiting time, queue length, and throughput.
- Emergency-vehicle priority and pedestrian-aware signal handling.
- Campus-specific traffic data collection and testing.

## 🌍 Social Relevance

The proposed system explores how AI can help reduce unnecessary vehicle waiting, improve traffic flow, and support safer campus entrances. Its potential effectiveness must be evaluated through controlled experiments before real-world deployment.

## ⚠️ Limitations

- Detection performance depends on video quality, camera angle, and lighting.
- A single camera view may not represent the entire junction.
- Basic congestion estimates may not accurately measure queue length or waiting time.
- Signal recommendations are experimental and are not intended to control real traffic infrastructure.

## 👩‍💻 Author

Developed as an AI-based academic project exploring computer vision and intelligent traffic management.

## 📄 License

This project is intended for educational and research purposes. A license can be added to specify permissions for reuse and distribution.
