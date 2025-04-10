from flask import Flask, Blueprint, request, jsonify, render_template  # ✅ Import Flask
import torch
import pickle
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import wikipedia
from sentence_transformers import SentenceTransformer, util
# ✅ Define Blueprint
chatbot_bp = Blueprint("chatbot", __name__, template_folder="templates", static_folder="static")
@chatbot_bp.route("/")
def index():
    return render_template('chatbot/index.html')  # Use unique path
#✅ Load Model & Tokenizer Once
model_path = r"D:\NewCode\Chatbot"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
# ✅ Load Label Encoder
with open(r"D:\NewCode\Chatbot\label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
# ✅ Load CSV for Answers
csv_path = r"D:\NewCode\Chatbot\test_data.csv"
df = pd.read_csv(csv_path)
df["category"] = label_encoder.transform(df["category"])
category_to_answer = dict(zip(df["category"], df["answer"]))
# ✅ Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
# ✅ Function to Predict Category
def predict_category(question):
    encoded = tokenizer(question, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
    predicted_category = label_encoder.inverse_transform([prediction])[0]
    return prediction
# ✅ Function to Predict Answer
# def predict_answer(question):
#     predicted_category = predict_category(question)
#     answer = category_to_answer.get(predicted_category, "Sorry, I don't have an answer for that.")
#     return answer



# Load model once
embedder = SentenceTransformer("all-MiniLM-L6-v2")
df["embedding_tensor"] = df["question"].apply(lambda x: embedder.encode(x, convert_to_tensor=True))
def search_duckduckgo(query):
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(f"https://lite.duckduckgo.com/lite/?q={query}", headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")
    results = soup.find_all("a", class_="result-link")

    for result in results:
        text = result.text.strip()
        if len(text.split()) > 5:  # Skip titles like "Home | Facebook"
            return text
    return "Sorry, I couldn't find anything useful online."

def predict_answer(question):
    # Step 1: Semantic similarity
    question_embedding = embedder.encode(question, convert_to_tensor=True)
    all_embeddings = torch.stack(df["embedding_tensor"].tolist())
    scores = util.pytorch_cos_sim(question_embedding, all_embeddings)
    best_idx = scores.argmax().item()
    best_score = scores[0][best_idx].item()

    print(f"Best match score: {best_score}")
    if best_score > 0.6:
        return df.iloc[best_idx]["answer"]

    # Step 2: DuckDuckGo fallback
    duck_answer = search_duckduckgo(question)
    if duck_answer and "sorry" not in duck_answer.lower():
        return duck_answer

    # Step 3: Wikipedia fallback
    try:
        return wikipedia.summary(question, sentences=2)
    except:
        return "Sorry, I couldn't find a reliable answer."


# def predict_answer(question):
#     predicted_category = predict_category(question)
#     answer = category_to_answer.get(predicted_category, "")
#     # If answer is missing or too generic, fallback to internet
#     if not answer or "sorry" in answer.lower() or "I don't have" in answer.lower():
#         try:
#             summary = wikipedia.summary(question, sentences=2)
#             return summary
#         except Exception:
#             return "Sorry, I couldn't find an answer either."
    
#     # Optional: check if answer is mismatched with question (simple keyword matching)
#     if "summer" in answer.lower() and "winter" in question.lower():
#         try:
#             summary = wikipedia.summary(question, sentences=2)
#             return summary
#         except Exception:
#             return answer  # fallback failed; show local answer anyway
#     return answer
# ✅ API Endpoint for Predictions
@chatbot_bp.route("/predict", methods=["POST"])
def predict():
    data = request.json
    question = data.get("question", "").strip().lower()
    print(f"Received question: {question}")  # Debugging log
    answer = predict_answer(question)
    print(f"Predicted answer: {answer}")  # Debugging log
    return jsonify({"answer": answer})









# from flask import Blueprint, request, jsonify, render_template
# import requests
# from bs4 import BeautifulSoup
# chatbot_bp = Blueprint("chatbot", __name__, template_folder="templates", static_folder="static")
# @chatbot_bp.route("/")
# def index():
#     return render_template("chatbot/index.html")
# # Local Q&A database
# qa_data = {
#     "best crop for winter": "Wheat, barley, and mustard are ideal crops for winter.",
#     "how to prevent plant diseases": "Rotate crops, use resistant varieties, and apply appropriate fungicides.",
# }
# def search_duckduckgo(query):
#     headers = {"User-Agent": "Mozilla/5.0"}
#     res = requests.get(f"https://lite.duckduckgo.com/lite/?q={query}", headers=headers)
#     soup = BeautifulSoup(res.text, "html.parser")
#     results = soup.find_all("a", class_="result-link")

#     for result in results:
#         return result.text.strip()
#     return "Sorry, I couldn't find anything on that."
# @chatbot_bp.route("/predict", methods=["POST"])
# def predict():
#     data = request.json
#     question = data.get("question", "").strip().lower()
#     # Step 1: Check local answers
#     answer = qa_data.get(question)
#     # Step 2: If not found, fallback to DuckDuckGo
#     if not answer:
#         answer = search_duckduckgo(question)
#     return jsonify({"answer": answer})
