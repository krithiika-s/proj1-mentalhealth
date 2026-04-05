import streamlit as st
import joblib
import re
import string

# --- Function to clean user input ---
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return text

# --- Page Configuration ---
st.set_page_config(page_title="Mental Health Predictor", page_icon="🧠", layout="centered")

st.title("🧠 Mental Health Risk Prediction")
st.write("Analyze social media text to detect signs of Depression, Anxiety, or PTSD.")

# --- Load the Machine Learning Model ---
@st.cache_resource # This caches the model so it doesn't reload on every button click
def load_models():
    try:
        model = joblib.load('svm_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return model, vectorizer, True
    except FileNotFoundError:
        return None, None, False

model, vectorizer, is_loaded = load_models()

# Show a warning if the user hasn't run the Jupyter Notebook yet
if not is_loaded:
    st.error("⚠️ Model files missing! Please run your Jupyter Notebook to generate 'svm_model.pkl' and 'tfidf_vectorizer.pkl' first.")

# --- User Input Section ---
user_input = st.text_area("Enter a social media post to analyze:", height=150)

# --- Prediction Logic ---
if st.button("Predict Risk"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    elif not is_loaded:
        st.error("Cannot make a prediction because the machine learning model is missing.")
    else:
        # 1. Clean the text
        cleaned_input = clean_text(user_input)
        
        # 2. Convert text to numbers using the loaded TF-IDF vectorizer
        vectorized_input = vectorizer.transform([cleaned_input])
        
        # 3. Make the prediction
        prediction = model.predict(vectorized_input)[0]
        probabilities = model.predict_proba(vectorized_input)[0]
        max_prob = max(probabilities) * 100
        
        # 4. Display the results
        st.success("Analysis Complete!")
        
        # Color coding the output based on severity
        if prediction.lower() == 'normal':
            st.info(f"### Result: {prediction} (Confidence: {max_prob:.2f}%)")
        else:
            st.error(f"### Result: Flagged for {prediction} (Confidence: {max_prob:.2f}%)")
            st.write("⚠️ *Note: This tool is for educational purposes and is not a medical diagnosis.*")