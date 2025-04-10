import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW  
from sklearn.preprocessing import LabelEncoder
# Load dataset
print("Loading dataset...")
dataset_path = r"D:\NewCode\Chatbot\test_data.csv"
df = pd.read_csv(dataset_path)
print("Dataset loaded!")
print("Encoding category labels...")
label_encoder = LabelEncoder()
df["category"] = label_encoder.fit_transform(df["category"])
num_labels = len(label_encoder.classes_)  
print("Encoding complete!")
# Tokenizer
print("Initializing tokenizer...")
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
print("Tokenizer initialized!")
# Custom Dataset Class
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
train_dataset = CustomDataset(df, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
print("Initializing model...")
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=num_labels)
print("Model initialized!")
optimizer = AdamW(model.parameters(), lr=5e-5)
loss_function = torch.nn.CrossEntropyLoss()
num_epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Starting training...")
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1} started...")
    model.train()
    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = loss_function(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            print(f"Epoch {epoch+1}, Step {step}: Loss = {loss.item()}")
    print(f"Epoch {epoch+1} completed!")
# Save Model
print("Training finished. Saving model...")
model_save_path = r"D:\NewCode\Chatbot"
model.save_pretrained(model_save_path)
print("Model saved successfully!")
# Save Label Encoder
# import pickle
# with open("backend/models/label_encoder.pkl", "wb") as f:
#     pickle.dump(label_encoder, f)
# print("Label encoder saved!")