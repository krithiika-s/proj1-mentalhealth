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

<img width="1525" height="787" alt="image" src="https://github.com/user-attachments/assets/19a3bf78-638a-4683-99c6-0ef0cbc880c9" />

![WhatsApp Image 2026-04-05 at 12 16 04](https://github.com/user-attachments/assets/6a4743b0-0d55-47d3-ad59-c820cba90a62)
### 📊 Exploratory Data Analysis (EDA)
We performed extensive EDA to understand the distribution of mental health labels and text characteristics.

![Exploratory Data Analysis](eda_charts.jpg)

**Key Findings:**
* **Class Distribution:** The dataset contains a high volume of 'Normal' and 'Depression' cases, with 'Personality Disorder' being the minority class.
* **Text Length:** Most social media posts are under 1,000 characters, showing that the model needs to be effective at analyzing short-form text.

