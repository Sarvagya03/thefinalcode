import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
from collections import defaultdict
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy, mode
from bayes_opt import BayesianOptimization
import os

# 🎯 Load dataset with previous data
try:
    data = joblib.load("model_data.pkl")
    if not isinstance(data, defaultdict):
        data = defaultdict(list, data)
except FileNotFoundError:
    data = defaultdict(list)

# ✅ Load accuracy tracking file
accuracy_file = "accuracy_log.json"
if os.path.exists(accuracy_file):
    with open(accuracy_file, "r") as f:
        accuracy_data = json.load(f)
else:
    accuracy_data = {"correct": 0, "total": 0}

# ✅ Feature Extraction

def extract_features(sequence):
    """🔍 Extract numerical features with robust error handling."""
    sequence = np.array([float(x) for x in sequence if str(x).replace('.', '', 1).isdigit()])
    if len(sequence) == 0:
        return np.zeros(10)
    fourier = np.fft.fft(sequence).real[:5]
    mode_value = mode(sequence, keepdims=True).mode  # Ensuring array output
    return np.concatenate([
        [np.mean(sequence), np.std(sequence), np.median(sequence)],
        mode_value[:1] if mode_value.size > 0 else [0],  # Ensuring valid index
        [entropy(np.bincount(sequence.astype(int)) + 1) if len(sequence) > 0 else 0],
        fourier
    ])

# ✅ ML Model Optimization

def optimize_ml(n_estimators, max_depth):
    ml_model = XGBClassifier(n_estimators=int(n_estimators), max_depth=int(max_depth), random_state=42)
    X, y = [], []
    for key, values in data.items():
        if values:
            extracted_features = extract_features(list(key))
            if isinstance(extracted_features, np.ndarray) and extracted_features.shape[0] == 10:
                for value in values:
                    X.append(extracted_features)
                    y.append(int(value.split()[0]) if isinstance(value, str) else int(value))
    if not X or not y or len(X) != len(y):
        print("⚠️ Warning: X and y have mismatched lengths, skipping optimization.")
        return 0
    X = np.vstack(X).astype(np.float32)
    y = np.array(y, dtype=np.int32)
    ml_model.fit(X, y)
    return ml_model.score(X, y)

# ✅ Bayesian Optimization and Explicit ML Training
pbounds = {'n_estimators': (50, 500), 'max_depth': (5, 50)}
optimizer = BayesianOptimization(f=optimize_ml, pbounds=pbounds, random_state=42)
optimizer.maximize()

best_params = optimizer.max['params']
ml_model = XGBClassifier(n_estimators=int(best_params['n_estimators']),
                         max_depth=int(best_params['max_depth']),
                         random_state=42)

X, y = [], []
for key, values in data.items():
    if values:
        extracted_features = extract_features(list(key))
        if isinstance(extracted_features, np.ndarray) and extracted_features.shape[0] == 10:
            for value in values:
                X.append(extracted_features)
                y.append(int(value.split()[0]) if isinstance(value, str) else int(value))
if X and y and len(X) == len(y):
    X = np.vstack(X).astype(np.float32)
    y = np.array(y, dtype=np.int32)
    ml_model.fit(X, y)
    print("✅ ML Model trained successfully!")
else:
    print("⚠️ Warning: Not enough valid data for ML training.")

# ✅ AI Model (BiLSTM + Attention)

class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(input_dim, 1, bias=False)
    def forward(self, x):
        attn_weights = torch.softmax(self.attn(x), dim=1)
        return torch.sum(attn_weights * x, dim=1)

class BiLSTMPredictor(nn.Module):
    def __init__(self):
        super(BiLSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size=10, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
        self.attention = Attention(128)
        self.fc = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.attention(x)
        x = self.fc(x)
        return self.softmax(x)

ai_model = BiLSTMPredictor()
ai_optimizer = optim.Adam(ai_model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
if os.path.exists("ai_model.pth"):
    ai_model.load_state_dict(torch.load("ai_model.pth"))
    print("✅ AI Model Loaded from ai_model.pth")
else:
    print("⚠️ AI Model not found! Training a new one from model_data.pkl...")
    
    X_train, y_train = [], []
    for key, values in data.items():
        extracted_features = extract_features(list(key))
        if isinstance(extracted_features, np.ndarray) and extracted_features.shape[0] == 10:
            for value in values:
                X_train.append(extracted_features)
                y_train.append(int(value.split()[0]) if isinstance(value, str) else int(value))

    if X_train and y_train and len(X_train) == len(y_train):
        X_train = torch.tensor(np.vstack(X_train), dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)

        for epoch in range(10):  # Adjust epochs as needed
            ai_optimizer.zero_grad()
            outputs = ai_model(X_train.unsqueeze(1))
            loss = loss_fn(outputs, y_train)
            loss.backward()
            ai_optimizer.step()

        torch.save(ai_model.state_dict(), "ai_model.pth")
        print("✅ New AI Model trained and saved as ai_model.pth!")
    else:
        print("⚠️ Not enough data to retrain AI model. It will be created when more data is available.")

# ✅ Load & Fine-tune GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# ✅ Prediction Function
def predict_next_numbers(history):
    features = np.array([extract_features(history)]).reshape(1, -1)
    ml_predicted = list(map(int, ml_model.predict(features)[:4]))
    ml_probs = ml_model.predict_proba(features)[0]
    big_prob = sum(ml_probs[i] for i in range(len(ml_probs)) if i >= 5)
    small_prob = sum(ml_probs[i] for i in range(len(ml_probs)) if i < 5)
    return ml_predicted, big_prob, small_prob

# 🎯 Main Execution Loop
while True:
    history = list(map(int, input("\n📝 Enter previous numbers: ").split()))
    predicted_numbers, big_prob, small_prob = predict_next_numbers(history)
    print(f"🔮 Predictions: {predicted_numbers}")
    print(f"📊 Probabilities → BIG (>=5): {big_prob:.2%} | SMALL (<5): {small_prob:.2%}")
    actual_number = int(input("🎯 Enter the actual number that appeared: "))
    data[tuple(history)].append(actual_number)
    joblib.dump(data, "model_data.pkl")
    torch.save(ai_model.state_dict(), "ai_model.pth")
    if actual_number in predicted_numbers:
        accuracy_data["correct"] += 1
    accuracy_data["total"] += 1
    with open(accuracy_file, "w") as f:
        json.dump(accuracy_data, f)
    print(f"✅ Accuracy: {accuracy_data['correct']}/{accuracy_data['total']} ({(accuracy_data['correct']/accuracy_data['total']*100):.2f}%)")
