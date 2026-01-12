import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
import tempfile
import warnings
warnings.filterwarnings("ignore")

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="centered"
)

st.title("🌾 Crop Recommendation System")
st.caption("XGBoost-based ML model with realistic farming inputs")

# -------------------------------
# LOAD MODEL FILES
# -------------------------------
@st.cache_resource
def load_model():
    with open("xgb_crop_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    with open("feature_names.pkl", "rb") as f:
        features = pickle.load(f)
    return model, encoder, features

try:
    model, encoder, features = load_model()
except FileNotFoundError:
    st.error("❌ Model files not found. Upload .pkl files.")
    st.stop()

# -------------------------------
# USER INPUTS (SLIDERS)
# -------------------------------
st.subheader("🔢 Enter Soil & Climate Parameters")

N = st.slider("Nitrogen (N) kg/ha", 0, 140, 50)
P = st.slider("Phosphorus (P) kg/ha", 0, 140, 40)
K = st.slider("Potassium (K) kg/ha", 0, 205, 40)
temperature = st.slider("Temperature (°C)", 5.0, 45.0, 25.0)
humidity = st.slider("Humidity (%)", 10.0, 100.0, 70.0)
ph = st.slider("Soil pH", 3.5, 9.5, 6.5)
rainfall = st.slider("Rainfall (mm)", 0.0, 300.0, 100.0)

input_dict = {
    'N': N,
    'P': P,
    'K': K,
    'temperature': temperature,
    'humidity': humidity,
    'ph': ph,
    'rainfall': rainfall
}

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("🌱 Recommend Crop"):
    user_input = np.array([input_dict[f] for f in features]).reshape(1, -1)

    predicted_class = model.predict(user_input)
    predicted_proba = model.predict_proba(user_input)

    crop_name = encoder.inverse_transform(predicted_class)[0]
    confidence = predicted_proba[0][predicted_class[0]] * 100

    st.success(f"🌾 Recommended Crop: **{crop_name.upper()}**")
    st.info(f"📊 Confidence: **{confidence:.2f}%**")

    # -------------------------------
    # TOP 3 CROPS
    # -------------------------------
    st.subheader("🔝 Top 3 Crop Recommendations")

    top_3_idx = np.argsort(predicted_proba[0])[-3:][::-1]
    top_crops = [
        (encoder.inverse_transform([i])[0], predicted_proba[0][i] * 100)
        for i in top_3_idx
    ]

    for i, (crop, prob) in enumerate(top_crops, 1):
        st.write(f"{i}. **{crop}** — {prob:.2f}%")

    # Bar Chart
    crops, probs = zip(*top_crops)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=list(crops), y=list(probs), palette="crest", ax=ax)
    ax.set_ylabel("Confidence (%)")
    ax.set_xlabel("Crop")
    ax.set_title("Top 3 Crop Recommendations")
    st.pyplot(fig)

    # -------------------------------
    # PDF REPORT GENERATION
    # -------------------------------
    def generate_pdf():
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        doc = SimpleDocTemplate(temp_file.name, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("<b>Crop Recommendation Report</b>", styles['Title']))
        story.append(Spacer(1, 12))

        story.append(Paragraph("<b>Input Parameters</b>", styles['Heading2']))
        for k, v in input_dict.items():
            story.append(Paragraph(f"{k} : {v}", styles['Normal']))

        story.append(Spacer(1, 12))
        story.append(Paragraph("<b>Prediction Result</b>", styles['Heading2']))
        story.append(Paragraph(f"Recommended Crop: <b>{crop_name}</b>", styles['Normal']))
        story.append(Paragraph(f"Confidence: {confidence:.2f}%", styles['Normal']))

        story.append(Spacer(1, 12))
        story.append(Paragraph("<b>Top 3 Crops</b>", styles['Heading2']))
        for crop, prob in top_crops:
            story.append(Paragraph(f"{crop}: {prob:.2f}%", styles['Normal']))

        doc.build(story)
        return temp_file.name

    pdf_path = generate_pdf()

    with open(pdf_path, "rb") as f:
        st.download_button(
            label="📄 Download Prediction Report (PDF)",
            data=f,
            file_name="crop_recommendation_report.pdf",
            mime="application/pdf"
        )
