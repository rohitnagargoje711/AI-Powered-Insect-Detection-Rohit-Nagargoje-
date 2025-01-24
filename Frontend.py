
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import google.generativeai as genai
from googletrans import Translator

# Function to configure the GenAI API
def configure_genai(api_key):
    if not api_key:
        raise ValueError("API key is missing. Provide it as a parameter.")
    genai.configure(api_key=api_key)

# Image preprocessing function
IMG_SIZE = (128, 128)  # Resize to the size used during training
def preprocess_image(image):
    img = image.resize(IMG_SIZE)
    img_array = img_to_array(img) / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0)

# Generate report for the pest
def generate_report(predicted_class_label, image_path):
    prompt = f"""
    Generate a concise report for farmers about the identified pest: {predicted_class_label}.
    Include the pest's lifecycle, damage it causes, and suggested actions for control and prevention.
    """
    organ = Image.open(image_path)
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([prompt, organ])
        return response.text
    except Exception as e:
        return f"Error during report generation: {e}"

# Streamlit App Configuration
st.set_page_config(
    page_title="Insect Detection Dashboard",
    page_icon="🐞",
    layout="wide"
)

# Load the trained model
@st.cache_resource
def load_trained_model():
    return load_model('Insect_detection.h5')  # Ensure this path is correct

model = load_trained_model()

# Sidebar enhancements
st.sidebar.title("🌿 Farmer's Assistant")
st.sidebar.markdown("### Explore Features:")
st.sidebar.write("Select options to customize your experience.")

# Language selection (Custom order: English, Marathi, Hindi, Japanese, Gujarati)
supported_languages = {
    "English": "en",
    "Marathi": "mr",
    "Hindi": "hi",
    "Japanese": "ja",
    "Gujarati": "gu",
    "Assamese": "as",
    "Bengali": "bn",
    "Bodo": "brx",
    "Dogri": "doi",
    "Gujarati": "gu",
    "Kannada": "kn",
    "Kashmiri": "ks",
    "Konkani": "gom",
    "Maithili": "mai",
    "Malayalam": "ml",
    "Manipuri (Meitei)": "mni",
    "Nepali": "ne",
    "Odia": "or",
    "Punjabi": "pa",
    "Sanskrit": "sa",
    "Santali": "sat",
    "Sindhi": "sd",
    "Tamil": "ta",
    "Telugu": "te",
    "Urdu": "ur"
}

language = st.sidebar.selectbox(
    "🌐 Select Language:",
    list(supported_languages.keys())
)

# Helpful tips for farmers
st.sidebar.markdown("### 🛠 Tips for Farmers:")
st.sidebar.info(
    """
    - Ensure the image is clear and focused.
    - Upload images in JPG, JPEG, or PNG format.
    - Identify pests early to avoid damage.
    - Follow pest control recommendations carefully.
    """
)

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.caption("Powered by AI & Streamlit")

# Main App Content
st.title("🐞 Insect Detection Dashboard")
st.write("Upload an insect image to classify it and get pest management recommendations.")
st.divider()

# File uploader
uploaded_file = st.file_uploader("Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

# API Key setup
API_KEY = "AIzaSyBH7IG7WX8Dov_3Yegv_32KtbGy5YoBCEQ"
configure_genai(API_KEY)

if uploaded_file:
    # Display uploaded image
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    with col2:
        if st.button("Predict"):
            with st.spinner("Predicting..."):
                # Preprocess and make predictions
                image = load_img(uploaded_file)
                processed_image = preprocess_image(image)
                prediction = model.predict(processed_image)
                
                # Class labels
                class_labels = ['Africanized Honey Bees (Killer Bees)', 'Aphids', 'Armyworms', 'Brown Marmorated Stink Bugs', 
                'Cabbage Loopers', 'Citrus Canker', 'Colorado Potato Beetles', 'Corn Borers', 'Corn Earworms', 
                'Fall Armyworms', 'Fruit Flies', 'Spider Mites', 'Thrips', 'Tomato Hornworms', 'Western Corn Rootworms']
                predicted_class_index = np.argmax(prediction, axis=1)[0]
                predicted_label = class_labels[predicted_class_index]
                confidence = np.max(prediction)

                if confidence >= 0.5:
                    # Display results
                    st.success(f"Prediction: *{predicted_label}*")
                    st.info(f"Confidence: *{confidence:.2%}*")
                    
                    # Fetch pest management report from Gemini API
                    report_data = generate_report(predicted_label, uploaded_file)
                    
                    # Translate if needed
                    if language != "English":
                        translator = Translator()
                        dest_lang = supported_languages[language]
                        report_data = translator.translate(report_data, dest=dest_lang).text
                    
                    st.write("### Pest Management Recommendations")
                    st.write(report_data)
                else:
                    st.warning("Confidence is below the selected threshold. Please review the image or adjust the threshold.")
