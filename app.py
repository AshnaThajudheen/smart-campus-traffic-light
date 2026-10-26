import streamlit as st
import cv2
import tempfile
import os
import pandas as pd
from collections import Counter
from ultralytics import YOLO
import subprocess
import imageio_ffmpeg

st.set_page_config(
    page_title="Smart Campus Traffic",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 Smart Campus Traffic Light System")
st.caption("AI-based vehicle detection and adaptive signal simulation")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

VEHICLES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

uploaded_video = st.file_uploader(
    "Upload a traffic video",
    type=["mp4", "avi", "mov", "mkv"]
)

confidence = st.slider(
    "Detection confidence",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05
)

if uploaded_video and st.button("Analyze Traffic", type="primary"):
    temp_input = tempfile.NamedTemporaryFile(
        delete=False, suffix=".mp4"
    )
    temp_input.write(uploaded_video.read())
    temp_input.close()

    cap = cv2.VideoCapture(temp_input.name)

    if not cap.isOpened():
        st.error("Could not open this video.")
        os.unlink(temp_input.name)
        st.stop()

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_file = tempfile.NamedTemporaryFile(
        delete=False, suffix=".mp4"
    )
    output_path = output_file.name
    output_file.close()

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    records = []
    progress = st.progress(0)
    status = st.empty()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result = model.predict(
                frame,
                conf=confidence,
                classes=list(VEHICLES.keys()),
                imgsz=640,
                verbose=False
            )[0]

            counts = Counter(
                VEHICLES[int(c)]
                for c in result.boxes.cls.cpu().numpy()
            )

            total = sum(counts.values())
            annotated = result.plot()

            cv2.putText(
                annotated,
                f"Vehicles: {total}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            writer.write(annotated)

            records.append({
                "Time (seconds)": round(frame_index / fps, 2),
                "Cars": counts["Car"],
                "Motorcycles": counts["Motorcycle"],
                "Buses": counts["Bus"],
                "Trucks": counts["Truck"],
                "Total": total
            })

            frame_index += 1

            if total_frames > 0:
                progress.progress(
                    min(frame_index / total_frames, 1.0)
                )

            if frame_index % 20 == 0:
                status.write(
                    f"Analyzing frame {frame_index}..."
                )

    finally:
        cap.release()
        writer.release()

    try:
        df = pd.DataFrame(records)

        if df.empty:
            st.error("No frames were processed.")
            st.stop()

        # Estimate typical detected traffic in the video.
        # These are detections per frame, not unique vehicles.
        average_count = float(df["Total"].mean())
        peak_count = int(df["Total"].max())

        if average_count < 3:
            congestion = "Low"
            signal = "Green"
            green_seconds = 10
        elif average_count < 8:
            congestion = "Medium"
            signal = "Green"
            green_seconds = 20
        else:
            congestion = "High"
            signal = "Green"
            green_seconds = 30

        st.success("Traffic analysis complete!")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Average detections/frame", f"{average_count:.1f}")
        c2.metric("Peak detections/frame", peak_count)
        c3.metric("Estimated congestion", congestion)
        c4.metric("Suggested green duration", f"{green_seconds} sec")

        st.subheader("🚦 Simulated signal recommendation")
        st.info(
            f"Traffic level: {congestion}. "
            f"Suggested green duration: {green_seconds} seconds."
        )
        st.caption(
            "This first version analyzes one camera view and recommends "
            "a signal duration. It does not yet compare multiple lanes "
            "or run a complete traffic-light cycle."
        )

        st.subheader("🎥 Vehicle detection output")
        # Convert the OpenCV output to browser-compatible H.264 MP4
converted_path = output_path.replace(".mp4", "_browser.mp4")

ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

result = subprocess.run(
    [
        ffmpeg_path,
        "-y",
        "-i", output_path,
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        converted_path
    ],
    capture_output=True,
    text=True
)

if result.returncode == 0 and os.path.exists(converted_path):
    with open(converted_path, "rb") as video_file:
        st.video(video_file.read())
else:
    st.warning("Video conversion failed. Showing the original output.")
    st.video(output_path)

if os.path.exists(converted_path):
    os.remove(converted_path)

        st.subheader("📊 Vehicle counts over time")
        st.line_chart(
            df.set_index("Time (seconds)")["Total"]
        )

        st.subheader("Detection data")
        st.dataframe(df.head(100), use_container_width=True)

        st.download_button(
            "Download detection report CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="traffic_report.csv",
            mime="text/csv"
        )

    finally:
        for path in [temp_input.name, output_path]:
            if os.path.exists(path):
                os.unlink(path)
