# predict_local.py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# ==== CONFIG ====
MODEL_PATH = "resnet50_plantvillage_final.h5"   # change if needed
IMG_SIZE = (224, 224)  # 👈 must match the trained model input size
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch',
    'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# ==== LOAD MODEL ====
print(f"Loading model from {MODEL_PATH} ...")
model = keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded successfully. Expected input:", model.input_shape)

# ==== PREDICT FUNCTION ====
def predict_image(img_path, model, class_names, target_size=IMG_SIZE, top_k=5, show=True):
    img = image.load_img(img_path, target_size=target_size)
    arr = image.img_to_array(img)
    x = np.expand_dims(arr, axis=0)
    x = preprocess_input(x)  # important: match training pipeline

    preds = model.predict(x)
    top_idx = np.argsort(preds[0])[-top_k:][::-1]
    top_probs = preds[0][top_idx]

    if show:
        plt.imshow(img)
        plt.axis("off")
        title = "\n".join([f"{class_names[i]}: {top_probs[j]:.4f}" for j, i in enumerate(top_idx)])
        plt.title(title, fontsize=10)
        plt.show()

    return [(class_names[i], float(top_probs[j])) for j, i in enumerate(top_idx)]

# ==== EXAMPLE USAGE ====
if __name__ == "__main__":
    sample_image = "00e0a4ab-ecbd-4560-a71c-b19d86bb087c___FREC_Pwd.M 4917.JPG"  # 👈 replace with your local image path
    results = predict_image(sample_image, model, CLASS_NAMES, IMG_SIZE)
    print("\nTop Predictions:")
    for cls, prob in results:
        print(f"{cls}: {prob:.4f}")
