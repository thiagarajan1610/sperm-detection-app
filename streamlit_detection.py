# ============================================
# STREAMLIT GUI FOR SPERM DETECTION (DOT OUTPUT)
# ============================================

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os

# --------------------------------------------
# 1. LOAD MODEL
# --------------------------------------------
@st.cache_resource
def load_model():
    model_path = r"best.pt"
    return YOLO(model_path)

model = load_model()

# --------------------------------------------
# 2. UI HEADER
# --------------------------------------------
st.title("🔬 Sperm Detection System")
st.write("Upload an image to detect sperm (Green = Normal, Red = Abnormal)")

# --------------------------------------------
# 3. FILE UPLOAD
# --------------------------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# --------------------------------------------
# 4. PROCESS IMAGE
# --------------------------------------------
if uploaded_file is not None:

    # Convert file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    original = image.copy()

    # Run detection
    results = model(image)[0]

    # Draw dots
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if conf < 0.4:
            continue

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        if cls == 1:
            color = (0, 255, 0)   # GREEN (Normal)
        else:
            color = (0, 0, 255)   # RED (Abnormal)

        cv2.circle(image, (cx, cy), 5, color, -1)

    # Convert BGR → RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    # --------------------------------------------
    # 5. DISPLAY
    # --------------------------------------------
    st.subheader("Original Image")
    st.image(original_rgb, use_container_width=True)

    st.subheader("Detection Output")
    st.image(image_rgb, use_container_width=True)

    # --------------------------------------------
    # 6. DOWNLOAD BUTTON
    # --------------------------------------------
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        cv2.imwrite(tmp.name, image)

        with open(tmp.name, "rb") as file:
            st.download_button(
                label="📥 Download Output Image",
                data=file,
                file_name="output.jpg",
                mime="image/jpeg"
            )

    os.remove(tmp.name)
