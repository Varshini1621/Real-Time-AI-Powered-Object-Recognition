import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model (Download yolov8n.pt if not present)
model = YOLO("yolov8n.pt")  # You can use "yolov8m.pt" for better accuracy

# Streamlit UI
st.title("📸🔍 Object Detection with YOLOv8")
st.sidebar.header("Upload an Image or Use Webcam")

# 📌 Upload Image Feature
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Read Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Perform Object Detection
    results = model(image)  # YOLO inference
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = model.names[int(box.cls[0])]
            color = (0, 255, 0)

            # Draw Bounding Box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Convert Image to RGB (for Streamlit)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display Image
    st.image(image, caption="Detected Objects", use_column_width=True)
    st.success("✅ Object detection completed!")

# 📌 Real-time Webcam Detection
if st.sidebar.button("Use Webcam"):
    st.warning("Press 'Q' to exit webcam mode.")

    cap = cv2.VideoCapture(0)  # Open webcam
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture webcam feed.")
            break

        results = model(frame)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                label = model.names[int(box.cls[0])]
                color = (0, 255, 0)

                # Draw Bounding Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("YOLOv8 Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'Q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()
