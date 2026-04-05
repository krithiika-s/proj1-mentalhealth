import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

# =========================
# Load dataset
# =========================
df = pd.read_csv('data.csv')  # make sure this file exists

# Convert labels to numbers
labels = list(df['label'].unique())
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

df['label_id'] = df['label'].map(label2id)

# Train-test split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'], df['label_id'], test_size=0.2, random_state=42
)

# =========================
# Tokenization
# =========================
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(
    list(train_texts),
    truncation=True,
    padding=True,
    max_length=128
)

val_encodings = tokenizer(
    list(val_texts),
    truncation=True,
    padding=True,
    max_length=128
)

# =========================
# Dataset class
# =========================
class MentalHealthDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels.iloc[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = MentalHealthDataset(train_encodings, train_labels)
val_dataset = MentalHealthDataset(val_encodings, val_labels)

# =========================
# Model
# =========================
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(labels)
)

# =========================
# Training setup
# =========================
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,   # keep 1 for faster training
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    logging_dir='./logs',
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# =========================
# Train
# =========================
trainer.train()

# =========================
# Save model
# =========================
model.save_pretrained('bert_model')
tokenizer.save_pretrained('bert_model')

print("✅ BERT model training complete and saved!")
