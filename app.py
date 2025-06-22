import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt
import json
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score
import seaborn as sns

# ----------------------
# ✅ Load the trained model FIRST (before calling evaluate_model)
model = tf.keras.models.load_model(os.path.join("model", "pneumonia_cnn_model.h5"))

# ----------------------
# ✅ Evaluate Model Function (cached)
@st.cache_resource
def evaluate_model(model):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator


    test_dir = "test"
    IMG_SIZE = 150
    BATCH_SIZE = 32

    test_datagen = ImageDataGenerator(rescale=1./255)

    test_data = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    y_true = test_data.classes
    y_pred = model.predict(test_data)
    y_pred_classes = (y_pred > 0.5).astype("int32").flatten()

    return y_true, y_pred_classes, test_data.class_indices

# ----------------------
# ✅ Streamlit UI
st.set_page_config(page_title="Pneumonia Detector", layout="centered")
st.title("🩻 Pneumonia Detection from Chest X-Ray")

uploaded_file = st.file_uploader("Upload an X-ray image...", type=["jpg", "jpeg", "png"])
reset = st.button("🔄 Clear")

# Clear uploaded image
if reset:
    uploaded_file = None
    st.rerun()

# Preprocess function
def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (150, 150))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("🧠 Predicting..."):
        img = preprocess_image(image)
        prediction = model.predict(img)[0][0]
        percent = round(prediction * 100, 2)

    if prediction > 0.5:
        st.error(f"⚠️ Prediction: **Pneumonia** ({percent}%)")
    else:
        st.success(f"✅ Prediction: **Normal** ({100 - percent}%)")

# ----------------------
# ✅ Show Evaluation Metrics
st.subheader("📉 Model Evaluation on Test Dataset")

y_true, y_pred, labels = evaluate_model(model)

# Confusion matrix and metrics
cm = confusion_matrix(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)

st.write(f"✅ **Accuracy**: {round(acc * 100, 2)}%")
st.write(f"🎯 **Precision**: {round(prec * 100, 2)}%")
st.write(f"🔁 **Recall**: {round(rec * 100, 2)}%")

# Display confusion matrix
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
ax_cm.set_title("🧾 Confusion Matrix")
st.pyplot(fig_cm)
