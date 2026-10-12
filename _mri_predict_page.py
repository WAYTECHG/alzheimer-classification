import streamlit as st
import numpy as np
import os
import joblib
from PIL import Image, ImageOps

def mri_predict_page():
    # Load the trained model
    def load_best_model(weights_path='models/alzheimer_mri.pkl'):
        try:
            model = joblib.load(weights_path)
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None

    def preprocess_single_image(image):
        # Load the image
        image = ImageOps.fit(image, (178,208), Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        image_array = image_array.astype(np.float32)
        data = np.ndarray(shape=(1,178,208,3), dtype=np.float32)

        return data

    # Function to preprocess and predict
    def preprocess_and_predict(image, model):
        try:
            image = preprocess_single_image(image)
            prediction = model.predict(image)
            return prediction
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            return None

    # Streamlit app
    st.title("Alzheimer's Disease Prediction from MRI")
    st.header("Upload an MRI Image")

    uploaded_file = st.file_uploader("Choose an MRI image...", type="jpg")

    if uploaded_file is not None:
        with st.spinner("Processing..."):
            # image_path = f"./temp_{uploaded_file.name}"
            # with open(image_path, "wb") as f:
            #     f.write(uploaded_file.getbuffer())
            image = Image.open(uploaded_file).convert('RGB')

            st.image(image, caption='Uploaded MRI Image', width=300)

            model = load_best_model()  # Load the model here to ensure it's available when needed
            if model is not None:
                prediction = preprocess_and_predict(image, model)
                if prediction is not None:
                    predicted_class = np.argmax(prediction, axis=1)[0]

                    classes = ['Mild Dementia', 'Moderate Dementia', 'Non Dementia', 'Very Mild Dementia']
                    predicted_label = classes[predicted_class]

                    st.write(f"Predicted Clinical Condition: **{predicted_label}**")

            # os.remove(image)

# if __name__ == "__main__":
#     mri_predict_page()
# mri_predict_page()