# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 15:49:00 2025

@author: 18452
"""

# -*- coding: utf-8 -*-
"""
RT-DETR + ByteTrack Live Tracking in Streamlit
Using Supervision library for elegant tracking and annotation.
"""

import streamlit as st
import torch
import cv2
import numpy as np
import time
from PIL import Image
from transformers import RTDetrImageProcessor, RTDetrForObjectDetection
import supervision as sv

# -------------------------------------
# 🧠 Load RT-DETR Model (Cached)
# -------------------------------------
@st.cache_resource
def load_model():
    st.write("🔄 Loading RT-DETR model...")
    processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")
    model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    st.success(f"✅ Model loaded on {device.upper()}")
    return processor, model, device

processor, model, device = load_model()

# -------------------------------------
# 🧩 Initialize Supervision Components
# -------------------------------------
@st.cache_resource
def init_supervision_components():
    """Initialize tracker and annotators"""
    tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=30
    )
    
    box_annotator = sv.BoundingBoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.5)
    trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=50)
    
    st.success("✅ Supervision components initialized")
    return tracker, box_annotator, label_annotator, trace_annotator

tracker, box_annotator, label_annotator, trace_annotator = init_supervision_components()

# -------------------------------------
# 🎨 Streamlit UI
# -------------------------------------
st.set_page_config(page_title="RT-DETR + ByteTrack Live Tracker", layout="wide")
st.title("⚡ RT-DETR + ByteTrack Live Object Tracking")
st.write("Real-time object detection & tracking using RT-DETR and Supervision library.")

col1, col2 = st.columns(2)
with col1:
    confidence = st.slider("Detection confidence threshold", 0.1, 1.0, 0.5, 0.05)
with col2:
    show_trace = st.checkbox("Show trace paths", value=True)

run = st.checkbox("🎥 Start Tracking", value=False)
frame_placeholder = st.image([])
stats_placeholder = st.empty()

# -------------------------------------
# 🎥 Find a Working Camera
# -------------------------------------
def find_working_camera():
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            st.success(f"✅ Camera index {i} works")
            cap.release()
            return i
        else:
            st.warning(f"❌ Camera index {i} not available")
    st.error("🚫 No working camera found.")
    return None

# -------------------------------------
# 🔄 Convert RT-DETR output to Supervision Detections
# -------------------------------------
def rtdetr_to_detections(results, model_config):
    """Convert RT-DETR results to Supervision Detections format"""
    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    labels = results["labels"].cpu().numpy()
    
    if len(boxes) == 0:
        return sv.Detections.empty()
    
    # Convert to xyxy format (already in this format from RT-DETR)
    xyxy = boxes
    
    # Create Supervision Detections object
    detections = sv.Detections(
        xyxy=xyxy,
        confidence=scores,
        class_id=labels.astype(int)
    )
    
    return detections

# -------------------------------------
# 🔄 Main Loop
# -------------------------------------
if run:
    cam_index = find_working_camera()
    if cam_index is not None:
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        cap.set(3, 640)
        cap.set(4, 480)

        fps_window = []
        frame_count = 0

        while run:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Unable to access webcam.")
                break

            frame_count += 1

            # Convert to RGB for RT-DETR
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)

            # Run RT-DETR detection
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)

            target_sizes = torch.tensor([image.size[::-1]]).to(device)
            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=confidence
            )[0]

            # Convert to Supervision Detections
            detections = rtdetr_to_detections(results, model.config)

            # Update tracker with detections
            detections = tracker.update_with_detections(detections)

            # Prepare labels for annotation
            labels = []
            if detections.tracker_id is not None:
                for i, (class_id, tracker_id, conf) in enumerate(
                    zip(detections.class_id, detections.tracker_id, detections.confidence)
                ):
                    class_name = model.config.id2label.get(int(class_id), "Unknown")
                    label = f"#{tracker_id} {class_name} {conf:.2f}"
                    labels.append(label)
            else:
                for i, (class_id, conf) in enumerate(zip(detections.class_id, detections.confidence)):
                    class_name = model.config.id2label.get(int(class_id), "Unknown")
                    label = f"{class_name} {conf:.2f}"
                    labels.append(label)

            # Annotate frame (work with BGR for OpenCV)
            annotated_frame = frame.copy()
            
            # Draw trace paths if enabled
            if show_trace and detections.tracker_id is not None:
                annotated_frame = trace_annotator.annotate(
                    scene=annotated_frame, 
                    detections=detections
                )
            
            # Draw bounding boxes
            annotated_frame = box_annotator.annotate(
                scene=annotated_frame, 
                detections=detections
            )
            
            # Draw labels
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, 
                detections=detections,
                labels=labels
            )

            # FPS calculation
            fps = 1 / (time.time() - start_time)
            fps_window.append(fps)
            if len(fps_window) > 20:
                fps_window.pop(0)
            avg_fps = sum(fps_window) / len(fps_window)
            
            # Add FPS text
            cv2.putText(
                annotated_frame, 
                f"FPS: {avg_fps:.1f}", 
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )

            # Convert back to RGB for Streamlit display
            display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(display_frame, use_container_width=True)
            
            # Display stats
            num_tracked = len(detections.tracker_id) if detections.tracker_id is not None else 0
            stats_placeholder.metric(
                label="Tracked Objects", 
                value=num_tracked,
                delta=f"Frame {frame_count}"
            )

        cap.release()
        st.success("Camera stopped.")