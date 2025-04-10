import torch
import pandas as pd
import pickle
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report
# Load the trained model
model_path = r"D:\NewCode\Chatbot"
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()  
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
with open("backend/models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
test_dataset_path = r"D:\NewCode\Chatbot\test_data.csv"  
df_test = pd.read_csv(test_dataset_path)
df_test["category"] = label_encoder.transform(df_test["category"])  
class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.texts = dataframe["question"].tolist()
        self.labels = dataframe["category"].tolist()
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx], padding="max_length", truncation=True, max_length=128, return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoded.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
test_dataset = CustomDataset(df_test, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
all_preds = []
all_labels = []
with torch.no_grad(): 
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)  
        all_preds.extend(predictions.cpu().numpy()) 
        all_labels.extend(labels.cpu().numpy())  
accuracy = accuracy_score(all_labels, all_preds)
from sklearn.metrics import classification_report
unique_labels = sorted(set(all_labels) | set(all_preds))  
report = classification_report(all_labels, all_preds, labels=unique_labels, target_names=label_encoder.classes_[:len(unique_labels)])
print("Classification Report:\n", report)
print(f"Test Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)
