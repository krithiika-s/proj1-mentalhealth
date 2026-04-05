import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from src.preprocess import clean_text

df = pd.read_csv('data/data.csv')
df['text'] = df['text'].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = SVC(probability=True)
model.fit(X_train_tfidf, y_train)

preds = model.predict(X_test_tfidf)
print(classification_report(y_test, preds))

joblib.dump(model, 'models/svm_model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')
