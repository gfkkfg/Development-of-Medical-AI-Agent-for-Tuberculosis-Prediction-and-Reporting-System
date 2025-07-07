import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from dotenv import load_dotenv
import yagmail
import time
from PIL import Image
import cv2
import os
from fpdf import FPDF
import pandas as pd
import csv
import groq
import requests

# CSV file to store feedback
FEEDBACK_CSV = "radiologist_feedback.csv"
UPLOAD_DIR = "uploaded_images/"

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load API Key for Groq AI
load_dotenv()
api_key = os.getenv("GROQ_API")
email_password = os.getenv("EMAIL_PASSWORD")

if not api_key:
    st.error("API Key Missing! Check your .env file.")
    st.stop()

# Ensure feedback file exists
if not os.path.exists(FEEDBACK_CSV):
    with open(FEEDBACK_CSV, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Radiologist Name", "RadiologistgEmail", "Patient ID", "Patient Email", "Hospital", "Location", "Specialty",
                         "Workstation ID", "X-ray Filename", "AI Severity (%)", "Assessment", "False Positive?", "False Negative?", "Comments"])

# Model Path
MODEL_PATH = "InceptionV3_tuberculosis.keras"

# Email details
sender_email = "gfkkfg6161@gmail.com"

# Check if file exists before loading
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at: {MODEL_PATH}")
    st.stop()

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")

except Exception as e:
    st.error(f"Model Loading Failed: {e}")
    st.stop()

# Function to get next image filename (0.jpg, 1.jpg, ...)
def get_next_image_filename(directory):
    existing_files = [f for f in os.listdir(directory) if f.endswith(".jpg")]
    if not existing_files:
        return "0.jpg"
    
    existing_numbers = sorted([int(f.split(".")[0]) for f in existing_files])
    next_number = existing_numbers[-1] + 1
    return f"{next_number}.jpg"

# Function to preprocess image for model prediction
def preprocess_image(img):
    img = img.resize((320, 320))  # Match model input size
    img = img.convert("RGB")
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Expand dims for model
    return img_array

# Function to generate Grad-CAM heatmap
def generate_gradcam(img, model, last_conv_layer_name):
    img_array = preprocess_image(img)
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    heatmap = cv2.resize(heatmap[0], (320, 320))
    return heatmap

# Function to overlay heatmap on image
def overlay_heatmap(img, heatmap):
    img_array = np.array(img)
    heatmap = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    if len(img_array.shape) == 2 or img_array.shape[2] == 1:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    overlayed = cv2.addWeighted(img_array, 0.6, heatmap, 0.4, 0)
    
    return Image.fromarray(overlayed) 


# Function to predict severity and generate Grad-CAM
def predict_severity(img):
    processed_img = preprocess_image(img)
    predictions = model.predict(processed_img)
    severity_score = float(predictions[0][0]) * 100
    heatmap = generate_gradcam(img, model, "mixed10")  # Last conv layer of InceptionV3
    return severity_score, heatmap

# Function to fetch AI-generated TB treatment summary
def get_tb_treatment_summary(formatted_severity):
    client = groq.Client(api_key=api_key)  

    prompt = f"""
        Provide a well-structured, **detailed**, and **medically accurate** summary of the latest tuberculosis (TB) treatment guidelines from WHO and CDC, specifically for **{formatted_severity} TB**.
        Your response must include:
        1. **First-line treatment** options (medications, dosage, duration, and administration method).
        2. **Second-line treatment** (if applicable for drug-resistant TB, including specific antibiotics and combination therapies).
        3. **Monitoring & testing requirements** (sputum tests, liver function, imaging, drug resistance testing).
        4. **Possible side effects & risk factors** (common side effects for each drug and management strategies).
        5. **Dietary & lifestyle modifications** (foods to avoid, recommended nutrition, exercise, and sleep guidelines).
        6. **Precautionary & preventive measures** (isolation protocols, mask-wearing, public health guidelines, vaccination details).
        7. **Recovery expectations & long-term care** (follow-up checkups, potential complications, relapse prevention).
        8. **Alternative therapies or new research updates** (latest WHO recommendations, clinical trials, emerging treatments).
        9. Format the response in **clear sections with bullet points** for readability.

        Ensure the response is **concise yet information-rich**, avoiding unnecessary repetition. If multiple treatment options exist, mention their relative effectiveness and when each should be used.
    """

    max_retries = 3
    retry_delay = 5  

    for attempt in range(max_retries):
        try:
            with st.spinner("Fetching AI-generated treatment plan..."):
                response = client.chat.completions.create(
                    model="Llama-3.3-70B-Versatile", 
                    messages=[{"role": "user", "content": prompt}]
                )
            return response.choices[0].message.content.strip()
        except Exception as e:
            st.warning(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(retry_delay)

    return "Error: Unable to fetch treatment summary."

# Function to generate PDF report
def generate_pdf(uploaded_img, processed_img, formatted_severity, treatment_summary, radiologist_name, radiologist_email, patient_id, patient_email, radiologist_hospital, radiologist_location, radiologist_specialty, workstation_id):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Tuberculosis Severity Report", ln=True, align="C")
    pdf.ln(10)
    
    # Radiologist Information
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Radiologist Information", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 8, f"Name: {radiologist_name}\nEmail: {radiologist_email}\nHospital: {radiologist_hospital}\nLocation: {radiologist_location}\nSpecialty: {radiologist_specialty}\nWorkstation ID: {workstation_id}")
    pdf.ln(5)
    
    # Patient Information
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Patient Information", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 8, f"Patient ID: {patient_id}\nPatient Email: {patient_email}")
    pdf.ln(5)
    
    # Severity Score
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "AI Severity Analysis", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 10, f"Predicted Severity Level: {formatted_severity}%", ln=True)
    pdf.ln(5)
    
    # Save images temporarily
    uploaded_img_path = "uploaded_xray.jpg"
    processed_img_path = "processed_xray.jpg"
    uploaded_img.save(uploaded_img_path, format="JPEG")
    processed_img.save(processed_img_path, format="JPEG")
    
    # X-ray Images
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "X-ray Images", ln=True)
    pdf.image(uploaded_img_path, x=10, y=pdf.get_y(), w=90)
    pdf.image(processed_img_path, x=110, y=pdf.get_y(), w=90)
    pdf.ln(90)
    
    # Treatment Plan
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "AI-Powered Treatment Plan", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 8, treatment_summary)
    pdf.ln(5)
    
    # Save PDF
    pdf_output = os.path.join(os.getcwd(), "TB_Severity_Report.pdf")
    try:
        pdf.output(pdf_output)
        # Clean up temporary files
        if os.path.exists(uploaded_img_path):
            os.remove(uploaded_img_path)
        if os.path.exists(processed_img_path):
            os.remove(processed_img_path)
        return pdf_output
    except Exception as e:
        return None
    
# Function to send email with the PDF report
def send_email(pdf_report,receiver_email):
    try:
        yag = yagmail.SMTP(sender_email, email_password)
        yag.send(
            to=receiver_email,
            subject="TB Severity Report - Patient Analysis",
            contents="Attached is the tuberculosis severity analysis report. Please review and provide feedback.",
            attachments=pdf_report
        )
        return "Email sent successfully!"
    except Exception as e:
        return f"Failed to send email: {e}"

# Streamlit UI
st.title("Tuberculosis Severity Prediction & Reporting")
st.subheader("🔍 AI-Powered TB Detection")
st.write("Upload a chest X-ray to get an AI-generated severity score and treatment plan.")

# Radiologist Information
st.subheader("👩‍⚕️ Radiologist Details")
radiologist_name = st.text_input("Full Name")
radiologist_email = st.text_input("Email (Radiologist)")
patient_id = st.text_input("Patient ID")
patient_email = st.text_input("Email (Patient)")
radiologist_hospital = st.text_input("Hospital/Clinic Name")
radiologist_location = st.text_input("Location (City, Country)")
radiologist_specialty = st.selectbox("Specialty", ["Radiology", "Pulmonology", "General Medicine", "Other"])
workstation_id = st.text_input("Workstation ID (Optional)")

# X-ray Upload
st.subheader("📤 Upload Chest X-ray")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)

    if "current_image_filename" not in st.session_state:
        st.session_state["current_image_filename"] = get_next_image_filename(UPLOAD_DIR)
    
    image_filename = st.session_state["current_image_filename"]
    img_path = os.path.join(UPLOAD_DIR, image_filename)

    img.save(img_path)

    st.image(img, caption="Uploaded X-ray", use_container_width=True)

    severity, heatmap = predict_severity(img)
    formatted_severity = round(severity, 2) 
    st.write(f"Predicted Severity Level: {formatted_severity:.2f}%")

    overlayed_img = overlay_heatmap(img, heatmap)
    st.image(overlayed_img, caption="Grad-CAM Heatmap", use_container_width=True)

    treatment_summary = get_tb_treatment_summary(severity)
    st.subheader("AI-Powered Treatment Plan:")
    st.text_area("Treatment Summary", treatment_summary, height=300)

    pdf_report = generate_pdf(img, overlayed_img, formatted_severity, treatment_summary, radiologist_name, radiologist_email, patient_id, patient_email, radiologist_hospital, radiologist_location, radiologist_specialty, workstation_id)
    if pdf_report and os.path.exists(pdf_report):
        with open(pdf_report, "rb") as pdf_file:
            st.download_button(label="Download Report", data=pdf_file, file_name="TB_Severity_Report.pdf", mime="application/pdf")

    # Send email to radiologist
    email_status1 = send_email(pdf_report, radiologist_email)
    email_status2 = send_email(pdf_report, patient_email)

    # Feedback Collection
    st.subheader("📝 Radiologist Feedback")
    ai_prediction = st.number_input("AI Predicted TB Severity (%)", min_value=0.0, max_value=100.0, value=formatted_severity)
    radiologist_assessment = st.selectbox("Radiologist's Severity Assessment", ["Normal", "Mild", "Moderate", "Severe"])
    false_positive = st.radio("False Positive Observed?", ["Yes", "No"])
    false_negative = st.radio("False Negative Observed?", ["Yes", "No"])
    additional_comments = st.text_area("Additional Comments")

    if st.button("Submit Feedback"):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(FEEDBACK_CSV, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, radiologist_name, radiologist_email, patient_id, patient_email, radiologist_hospital, radiologist_location,
                             radiologist_specialty, workstation_id, image_filename, ai_prediction,
                             radiologist_assessment, false_positive, false_negative, additional_comments])
        st.success("Feedback Saved Successfully!")
