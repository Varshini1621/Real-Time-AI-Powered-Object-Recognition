import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load YOLO model
model = YOLO("yolov8n.pt")

st.title("📸🔍 Object Detection with YOLOv8")

st.sidebar.header("Upload Image or Use Camera")

# Upload Image
uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg","jpeg","png"])

# Camera Input
camera_image = st.camera_input("Take a picture")

image = None

# If user uploads image
if uploaded_file:
    image = Image.open(uploaded_file)

# If user captures photo
elif camera_image:
    image = Image.open(camera_image)

if image is not None:

    # Convert to numpy
    frame = np.array(image)

    # Run YOLO detection
    results = model(frame)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = model.names[int(box.cls[0])]

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,f"{label} {conf:.2f}",(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

    st.image(frame, caption="Detected Objects", use_container_width=True)
