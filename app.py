
import os
import tempfile
import subprocess
from collections import Counter

import cv2
import pandas as pd
import streamlit as st
import imageio_ffmpeg
from ultralytics import YOLO


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Smart Campus Traffic Light System",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 Smart Campus Traffic Light System")
st.caption(
    "AI-based vehicle detection and adaptive traffic signal simulation"
)


# --------------------------------------------------
# LOAD YOLO MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


try:
    model = load_model()
except Exception as e:
    st.error("Could not load the YOLO model.")
    st.exception(e)
    st.stop()


# COCO vehicle class IDs
VEHICLES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}


# --------------------------------------------------
# VIDEO UPLOAD
# --------------------------------------------------
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


# --------------------------------------------------
# PROCESS VIDEO
# --------------------------------------------------
if uploaded_video and st.button(
    "Analyze Traffic",
    type="primary"
):

    temp_input_path = None
    output_path = None
    converted_path = None
    cap = None
    writer = None

    try:
        # Save uploaded video
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ) as temp_file:
            temp_file.write(uploaded_video.getvalue())
            temp_input_path = temp_file.name

        cap = cv2.VideoCapture(temp_input_path)

        if not cap.isOpened():
            st.error(
                "Could not open the uploaded video. "
                "Try another MP4 video."
            )
            st.stop()

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 25.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if width <= 0 or height <= 0:
            st.error("The uploaded video has invalid dimensions.")
            st.stop()

        # Create output video
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ) as output_file:
            output_path = output_file.name

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        if not writer.isOpened():
            st.error("Could not create the output video.")
            st.stop()

        records = []
        progress = st.progress(0)
        status = st.empty()

        frame_index = 0

        # Process every frame
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            result = model.predict(
                source=frame,
                conf=confidence,
                classes=list(VEHICLES.keys()),
                imgsz=640,
                verbose=False
            )[0]

            counts = Counter()

            if result.boxes is not None:
                class_ids = (
                    result.boxes.cls.cpu().numpy().astype(int)
                )

                for class_id in class_ids:
                    if class_id in VEHICLES:
                        counts[VEHICLES[class_id]] += 1

            total = sum(counts.values())

            # Draw YOLO bounding boxes
            annotated = result.plot()

            cv2.putText(
                annotated,
                f"Vehicles detected: {total}",
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
                status.info(
                    f"Processing frame {frame_index}"
                    + (
                        f" of {total_frames}"
                        if total_frames > 0
                        else ""
                    )
                )

        cap.release()
        cap = None

        writer.release()
        writer = None

        progress.progress(1.0)
        status.success("Video processing completed!")

        # Check output
        if not records:
            st.error("No video frames were processed.")
            st.stop()

        df = pd.DataFrame(records)

        if (
            not os.path.exists(output_path)
            or os.path.getsize(output_path) == 0
        ):
            st.error("The output video file is empty.")
            st.stop()

        # --------------------------------------------------
        # CONGESTION ESTIMATION
        # --------------------------------------------------
        average_count = float(df["Total"].mean())
        peak_count = int(df["Total"].max())

        if average_count < 3:
            congestion = "Low"
            green_seconds = 10
        elif average_count < 8:
            congestion = "Medium"
            green_seconds = 20
        else:
            congestion = "High"
            green_seconds = 30

        st.success("Traffic analysis completed!")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Average detections/frame",
            f"{average_count:.1f}"
        )
        col2.metric(
            "Peak detections/frame",
            peak_count
        )
        col3.metric(
            "Estimated congestion",
            congestion
        )
        col4.metric(
            "Suggested green duration",
            f"{green_seconds} sec"
        )

        st.subheader("🚦 Adaptive Signal Recommendation")

        st.info(
            f"Estimated congestion: {congestion}. "
            f"Suggested green duration: {green_seconds} seconds."
        )

        st.caption(
            "This initial prototype analyzes a single camera view. "
            "It recommends a timing based on average detections; "
            "it does not yet measure real queue length, compare "
            "multiple lanes, or implement a complete signal cycle."
        )

        # --------------------------------------------------
        # CONVERT VIDEO FOR BROWSER PLAYBACK
        # --------------------------------------------------
        st.subheader("🎥 Vehicle Detection Output")

        converted_path = output_path.replace(
            ".mp4",
            "_browser.mp4"
        )

        try:
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

            conversion = subprocess.run(
                [
                    ffmpeg_path,
                    "-y",
                    "-i", output_path,
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    "-an",
                    converted_path
                ],
                capture_output=True,
                text=True,
                timeout=180
            )

            if (
                conversion.returncode == 0
                and os.path.exists(converted_path)
                and os.path.getsize(converted_path) > 0
            ):
                with open(converted_path, "rb") as video_file:
                    video_bytes = video_file.read()

                st.video(video_bytes, format="video/mp4")

            else:
                st.warning(
                    "Browser-compatible conversion failed. "
                    "Trying the original output video."
                )

                with open(output_path, "rb") as video_file:
                    st.video(video_file.read(), format="video/mp4")

                with st.expander("Video conversion details"):
                    st.code(conversion.stderr[-3000:])

        except Exception as video_error:
            st.warning(
                "The video was processed, but playback conversion failed."
            )
            st.exception(video_error)

        # --------------------------------------------------
        # CHART AND DATA
        # --------------------------------------------------
        st.subheader("📊 Vehicle Counts Over Time")

        st.line_chart(
            df.set_index("Time (seconds)")["Total"]
        )

        st.subheader("Vehicle Detection Report")

        st.dataframe(
            df.head(100),
            use_container_width=True
        )

        st.download_button(
            "⬇️ Download Detection Report (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="traffic_report.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error("An error occurred during traffic analysis.")
        st.exception(e)

    finally:
        if cap is not None:
            cap.release()

        if writer is not None:
            writer.release()

        for path in [
            temp_input_path,
            output_path,
            converted_path
        ]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.caption(
    "Smart Campus Traffic Light System | "
    "Educational AI prototype — not for direct control "
    "of real traffic infrastructure."
)
