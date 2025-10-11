import joblib
import numpy as np
from PIL import ImageOps, Image

# Function to predict
def predict(image):
    model = joblib.load("alzheimer_mri.pkl")

    image = ImageOps.fit(image, (178,208), Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    image_array = image_array.astype(np.float32)
    data = np.ndarray(shape=(1,178,208,3), dtype=np.float32)
    print("done")
    prediction = model.predict(data)
    print("done")
    predicted_class = np.argmax(prediction, axis=1)[0]

    classes = ['Mild Dementia', 'Moderate Dementia', 'Non Dementia', 'Very Mild Dementia']
    predicted_label = classes[predicted_class]
    print(predicted_label)
    return predicted_label

# file = "28.jpg"
# image = Image.open(file).convert('RGB')
# predict(image)