# rt-detr-bytetrack-streamlit
Streamlit app for real-time object detection &amp; tracking using RT-DETR and ByteTrack.

Real-time object detection and multi-object tracking using RT-DETR and ByteTrack, all wrapped inside an interactive Streamlit web app.
This project demonstrates a modern, high-performance computer vision pipeline with bounding boxes, tracking IDs, label overlays, and trace paths—running live on a webcam feed.

🚀 Features
🔍 RT-DETR Object Detection

Powered by PekingU/rtdetr_r50vd (HuggingFace Transformers)

High-quality, transformer-based object detection

Adjustable confidence threshold

GPU acceleration supported

🧭 ByteTrack Multi-Object Tracking

Stable and consistent tracking IDs

Long-term trajectory buffering

Handles occlusions and ID preservation

Optional movement trace visualization

🎨 Supervision-Based Annotation

Clean bounding boxes

Text labels (track ID + class name + confidence)

Optional trace lines showing object histories

High-quality visual overlays

🎥 Live Webcam Stream

Auto-detects a working camera

Displays annotated frames in real-time

Shows FPS, object count, and per-frame updates

🧰 Streamlit UI

Start/stop camera

Slider for confidence threshold

Enable/disable trace paths

Responsive live video panel
