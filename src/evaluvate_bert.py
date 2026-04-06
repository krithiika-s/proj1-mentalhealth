import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import BertTokenizer, BertForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Load dataset
# =========================
df = pd.read_csv('data.csv')

# Label encoding
labels = list(df['label'].unique())
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

df['label_id'] = df['label'].map(label2id)

# Split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'], df['label_id'], test_size=0.2, random_state=42
)

# =========================
# Load trained model
# =========================
tokenizer = BertTokenizer.from_pretrained('bert_model')
model = BertForSequenceClassification.from_pretrained('bert_model')

model.eval()

# =========================
# Predictions
# =========================
y_pred = []

for text in test_texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    pred = torch.argmax(logits, dim=1).item()
    y_pred.append(pred)

# =========================
# Metrics
# =========================
acc = accuracy_score(test_labels, y_pred)
print(f"\n✅ BERT Accuracy: {acc:.4f}")

print("\n📊 Classification Report:")
print(classification_report(test_labels, y_pred, target_names=labels))

# =========================
# Confusion Matrix
# =========================
cm = confusion_matrix(test_labels, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels,
            yticklabels=labels)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("BERT Confusion Matrix")

plt.savefig("results/bert_confusion_matrix.png")
print("\n📁 Saved: results/bert_confusion_matrix.png")
