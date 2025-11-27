from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from datasets import load_dataset
from sklearn.metrics import f1_score

# Загрузить данные
dataset = load_dataset("Davlan/sib200", lang="rus")

# Загрузить предобученную модель RuBERT для эмбеддингов
tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
model = AutoModel.from_pretrained("cointegrated/rubert-tiny")

def embed_texts(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings

# Получаем эмбеддинги для тренировочного и тестового датасетов
X_train = embed_texts(dataset["train"]["text"])
y_train = np.array(dataset["train"]["label"])
X_test = embed_texts(dataset["test"]["text"])
y_test = np.array(dataset["test"]["label"])

# Обучаем классический классификатор (например, логистическую регрессию)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Предсказания и оценка
y_pred = clf.predict(X_test)
print("F1 macro:", f1_score(y_test, y_pred, average="macro"))

