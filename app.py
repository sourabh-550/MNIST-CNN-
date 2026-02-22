import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="MNIST Digit Recognizer",
    page_icon="🧠",
    layout="wide"
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("📌 Project Info")
st.sidebar.markdown("""
**Model:** CNN  
**Dataset:** MNIST  
**Test Accuracy:** 99.36%  

### Instructions:
- Draw digit clearly
- Keep strokes moderate
- Stay near center
""")

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("outputs/mnist_cnn_model.h5")

model = load_model()

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.title("🖌️ Handwritten Digit Recognition")
st.write("Draw a digit below. Prediction updates automatically.")

# --------------------------------------------------
# LAYOUT
# --------------------------------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("✏️ Draw Here")

    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=10,   # 🔥 thinner stroke
        stroke_color="white",
        background_color="black",
        height=200,        # 🔥 smaller canvas
        width=200,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.subheader("📊 Prediction")

    result_area = st.empty()

# --------------------------------------------------
# SIMPLE PREPROCESSING
# --------------------------------------------------
def preprocess_image(img):

    img = Image.fromarray(img.astype("uint8"))
    img = img.convert("L")

    # Resize directly to 28x28
    img = img.resize((28, 28))

    img_array = np.array(img).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    return img_array

# --------------------------------------------------
# AUTO PREDICT
# --------------------------------------------------
if canvas_result.image_data is not None:

    processed = preprocess_image(canvas_result.image_data)

    prediction = model.predict(processed)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    with result_area.container():
        st.success(f"🎯 Predicted: {predicted_digit}")
        st.write(f"Confidence: {confidence:.2f}%")
        st.bar_chart(prediction[0])