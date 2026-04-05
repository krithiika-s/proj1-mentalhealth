# Mental Health Status Classification from Social Media Text 🧠💻
**Project #26 | Predictive Analytics Group Project (2025-26)**

## 👥 Team Members
* KRITHIKA S
* UMAPARVATHY C.S
* HARIKRISHNAN S.M

## 📌 Problem Statement & Motivation
Mental health conditions often go undiagnosed. Social media platforms like Reddit and Twitter provide a vast amount of user-generated text that can be analyzed for early detection of mental health risks. This project aims to classify social media posts into specific mental health categories (Depression, Anxiety, PTSD, Normal) to explore the potential for early intervention tools, while carefully considering data privacy and ethical boundaries.

## 📊 Dataset Description
* **Source:** [Specify your dataset, e.g., Kaggle Reddit Mental Health Dataset]
* **Features:** Raw social media text posts.
* **Classes:** Depression, Anxiety, PTSD, Normal.
* **Challenges:** Class imbalance and handling noisy text data (slang, emojis, URLs).

## ⚙️ Methodology & Life Cycle Stages
This project strictly follows the 10-stage Data Science Life Cycle:
1. **Problem Definition:** Outlined above.
2. **Data Collection:** Sourced text data from social media.
3. **Data Cleaning:** Removed URLs, @mentions, emojis, and stop-words. Applied lemmatization.
4. **EDA:** Word clouds, n-gram frequency analysis, and post-length distributions.
5. **Feature Engineering:** `TF-IDF` for the baseline model, Tokenization for BERT.
6. **Model Building:** * *Baseline:* Support Vector Machine (SVM)
   * *Advanced:* Fine-tuned BERT (Transformer)
7. **Evaluation:** Comparing Precision, Recall (Sensitivity), Specificity, and F1-scores.
8. **Explainability:** Using SHAP/LIME to highlight text features driving predictions.
9. **Deployment:** Interactive Streamlit web application.
10. **Documentation:** GitHub repository, README, and PPT presentation.

## 🚀 How to Run the Project Locally
1. Clone the repository: `git clone https://github.com/krithiika-s/proj1-mentalhealth.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Jupyter Notebook for EDA and Modeling.
4. Run the Streamlit app: `streamlit run app.py`

## 🌐 Live Streamlit Deployment
https://proj1-mentalhealth-4e45aes56up3zikxsmt7el.streamlit.app/

## 📸 Application Screenshots
*Screenshots of the live app and EDA charts will be added here.*
