# %%
# Config & imports
import collections
import re

import evaluate
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, concatenate_datasets, load_dataset
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    pipeline,
)

SEED = 42
MODEL_NAME = "DeepPavlov/rubert-base-cased"
DATASET_NAME = "Davlan/sib200"
DATASET_LANGUAGE = "rus_Cyrl"
MINIBATCH_SIZE = 8
MAX_LEN = 512

np.random.seed(SEED)
torch.manual_seed(SEED)


# %%
# Нормализация текста
def normalize_text(text):
    """
    Щадящая нормализация: Unicode NFKC + тримминг пробелов.
    Регистр и пунктуация сохраняются, что важно для cased-модели.
    """
    if not isinstance(text, str):
        return ""
    import unicodedata

    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_batch_texts(texts):
    return [normalize_text(text) for text in texts]


# %%
# Примитивные аугментации текста
def split_sentences(text):
    parts = re.split(r"([.!?])", text)
    sents = []
    for i in range(0, len(parts), 2):
        seg = parts[i].strip()
        if not seg:
            continue
        end = parts[i + 1] if i + 1 < len(parts) else ""
        sents.append((seg + end).strip())
    return [s for s in sents if s]


def aug_sentence_shuffle(text, p=0.35):
    if np.random.rand() > p:
        return text
    sents = split_sentences(text)
    if len(sents) < 2:
        return text
    np.random.shuffle(sents)
    return " ".join(sents)


def aug_punct_swap(text, p=0.25):
    if np.random.rand() > p:
        return text
    puncts = [".", ",", "!", "?"]
    chars = list(text)
    for i, ch in enumerate(chars):
        if ch in puncts and np.random.rand() < 0.2:
            chars[i] = np.random.choice(puncts)
    return "".join(chars)


def aug_truncate_middle(text, p=0.25):
    if np.random.rand() > p:
        return text
    words = text.split()
    n = len(words)
    if n < 12:
        return text
    cut = int(n * np.random.uniform(0.1, 0.3))
    start_keep = int((n - cut) / 2)
    new_words = words[:start_keep] + words[start_keep + cut :]
    return " ".join(new_words)


def apply_augmentations(text):
    text = aug_sentence_shuffle(text, p=0.4)
    text = aug_punct_swap(text, p=0.3)
    text = aug_truncate_middle(text, p=0.3)
    return text


# %%
# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# %%
# Load data
train_set = load_dataset(DATASET_NAME, DATASET_LANGUAGE, split="train")
validation_set = load_dataset(DATASET_NAME, DATASET_LANGUAGE, split="validation")
test_set = load_dataset(DATASET_NAME, DATASET_LANGUAGE, split="test")

# %%
# Нормализация текстов
print("Применяем нормализацию текстов...")


def normalize_dataset(dataset):
    normalized_texts = normalize_batch_texts(dataset["text"])
    return dataset.remove_columns(["text"]).add_column("text", normalized_texts)


train_set = normalize_dataset(train_set)
validation_set = normalize_dataset(validation_set)
test_set = normalize_dataset(test_set)

print(f"Пример нормализованного текста: {train_set[0]['text'][:100]}...")

# %%
# Анализ распределения классов
print("\n=== Анализ распределения классов ===")
train_labels = train_set["category"]
label_counts = collections.Counter(train_labels)

print(f"Всего классов: {len(label_counts)}")
print(f"Общее количество примеров: {len(train_labels)}")
print("\nРаспределение по классам:")
for label, count in label_counts.most_common():
    print(f"  {label}: {count} примеров ({count / len(train_labels) * 100:.2f}%)")

# %%
# Усиленный oversampling для редких классов (entertainment, geography)
counts = collections.Counter(train_set["category"])
rare_classes = {"entertainment", "geography"}
if len(counts) > 0:
    # Подтягиваем редкие классы примерно до 95% от самого частого
    target = int(max(counts.values()) * 0.95)
    aug_texts, aug_labels = [], []
    rng = np.random.default_rng(SEED)
    for label in counts:
        if label not in rare_classes:
            continue
        need = max(0, target - counts[label])
        if need == 0:
            continue
        idxs = [i for i, l in enumerate(train_set["category"]) if l == label]
        for _ in range(need):
            i = int(rng.choice(idxs))
            base_text = train_set[i]["text"]
            new_text = apply_augmentations(base_text)
            aug_texts.append(new_text)
            aug_labels.append(label)

    if aug_texts:
        print(
            f"Добавляем аугментированных примеров для редких классов: {len(aug_texts)} (целевой уровень ~{target})"
        )
        aug_ds = Dataset.from_dict({"text": aug_texts, "category": aug_labels})
        train_set = concatenate_datasets([train_set, aug_ds])
        train_set = train_set.shuffle(seed=SEED)

# %%
# Опциональная псевдоразметка для entertainment:
# берём часть train, где модель уверена, и добавляем в train_set.
# Для запуска первый раз используется базовый RuBERT.
USE_PSEUDO_LABELING = True
PSEUDO_FRACTION = 0.2  # до 20% train для псевдоразметки
PSEUDO_MIN_PROB = 0.80  # порог уверенности

if USE_PSEUDO_LABELING:
    print("\n=== Псевдоразметка части train для entertainment ===")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(set(train_set["category"])),
        classifier_dropout=0.1,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )

    tmp_tokenized = train_set.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=MAX_LEN),
        batched=True,
    )
    tmp_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

    base_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model.to(device)

    pseudo_texts = []
    pseudo_labels = []

    # Не размечаем всё, только фракцию
    n_pseudo_candidates = int(len(tmp_tokenized) * PSEUDO_FRACTION)
    idxs_for_pseudo = np.random.choice(
        len(tmp_tokenized), size=n_pseudo_candidates, replace=False
    )

    softmax = nn.Softmax(dim=-1)
    with torch.no_grad():
        for idx in idxs_for_pseudo:
            sample = tmp_tokenized[idx]
            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
            outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = softmax(logits)[0].cpu().numpy()
            pred_idx = int(np.argmax(probs))
            conf = float(probs[pred_idx])

            # Пытаемся найти "очевидные" entertainment
            if conf >= PSEUDO_MIN_PROB:
                label = base_model.config.id2label.get(pred_idx, None)
                if label == "entertainment":
                    pseudo_texts.append(train_set[idx]["text"])
                    pseudo_labels.append(label)

    if pseudo_texts:
        print(f"Добавляем псевдоразмеченных entertainment: {len(pseudo_texts)}")
        pseudo_ds = Dataset.from_dict({"text": pseudo_texts, "category": pseudo_labels})
        train_set = concatenate_datasets([train_set, pseudo_ds])
        train_set = train_set.shuffle(seed=SEED)

# %%
# Labels mapping
list_of_categories = sorted(
    list(
        set(train_set["category"])
        | set(validation_set["category"])
        | set(test_set["category"])
    )
)
indices_of_categories = list(range(len(list_of_categories)))
n_categories = len(list_of_categories)
id2label = dict(zip(indices_of_categories, list_of_categories))
label2id = dict(zip(list_of_categories, indices_of_categories))

print("\nСписок категорий:")
for i, c in enumerate(list_of_categories):
    print(i, c)


# %%
# Вычисление весов классов
def compute_class_weights(labels, label2id, method="sqrt"):
    counts_local = collections.Counter(labels)
    classes_in_order = [lbl for lbl, _ in sorted(label2id.items(), key=lambda x: x[1])]
    n_classes = len(classes_in_order)

    class_counts_by_idx = [counts_local.get(lbl, 0) for lbl in classes_in_order]

    if method == "balanced":
        total = sum(class_counts_by_idx)
        weights = [
            (total / (n_classes * c)) if c > 0 else 1.0 for c in class_counts_by_idx
        ]
    elif method == "inverse":
        max_c = max(class_counts_by_idx) if any(class_counts_by_idx) else 1
        weights = [(max_c / c) if c > 0 else 1.0 for c in class_counts_by_idx]
    elif method == "sqrt":
        max_c = max(class_counts_by_idx) if any(class_counts_by_idx) else 1
        weights = [np.sqrt(max_c / c) if c > 0 else 1.0 for c in class_counts_by_idx]
    elif method == "custom":
        # Базовый sqrt + ручной буст для entertainment / geography
        max_c = max(class_counts_by_idx) if any(class_counts_by_idx) else 1
        weights = [np.sqrt(max_c / c) if c > 0 else 1.0 for c in class_counts_by_idx]
        for i, lbl in enumerate(classes_in_order):
            if lbl == "entertainment":
                weights[i] *= 5.0
            if lbl == "geography":
                weights[i] *= 1.5
    else:
        raise ValueError(f"Unknown method: {method}")

    print("\nВеса классов:")
    for i, lbl in enumerate(classes_in_order):
        print(
            f"  idx={i}, label={lbl}, count={class_counts_by_idx[i]}, weight={weights[i]:.4f}"
        )

    return torch.tensor(weights, dtype=torch.float32)


print("\nВычисление весов классов для тренировочного набора...")
class_weights = compute_class_weights(train_set["category"], label2id, method="custom")


# %%
# Tokenize
def tok(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)


tokenized_train_set = train_set.map(tok, batched=True)
tokenized_validation_set = validation_set.map(tok, batched=True)

labeled_train_set = tokenized_train_set.add_column(
    "label", [label2id[val] for val in tokenized_train_set["category"]]
)
labeled_validation_set = tokenized_validation_set.add_column(
    "label", [label2id[val] for val in tokenized_validation_set["category"]]
)

# %%
# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

# %%
# Метрики
cls_metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    f1_macro = cls_metric.compute(
        predictions=predictions, references=labels, average="macro"
    )["f1"]
    accuracy = (predictions == labels).mean()
    return {"f1": f1_macro, "accuracy": accuracy}


# %%
# Model
classifier = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=n_categories,
    id2label=id2label,
    label2id=label2id,
    classifier_dropout=0.2,
    hidden_dropout_prob=0.2,
    attention_probs_dropout_prob=0.2,
)

# %%
# Training arguments — чуть меньше эпох, больше warmup, clip grad norm
training_args = TrainingArguments(
    output_dir="rubert_sib200_weighted_v2",
    learning_rate=2.5e-5,
    per_device_train_batch_size=MINIBATCH_SIZE,
    per_device_eval_batch_size=MINIBATCH_SIZE,
    gradient_accumulation_steps=2,
    num_train_epochs=10,
    weight_decay=0.05,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_steps=50,
    warmup_ratio=0.12,
    lr_scheduler_type="linear",
    seed=SEED,
    data_seed=SEED,
    report_to=["none"],
    fp16=torch.cuda.is_available(),
    no_cuda=not torch.cuda.is_available(),
    label_smoothing_factor=0.03,
    gradient_checkpointing=True,
    max_grad_norm=1.0,
)


# %%
# Кастомный Trainer с взвешенным лоссом
class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.get("logits")  # (batch, num_classes)

        # --- Focal Loss c class_weights ---
        # Параметры фокал-лосса (можно поэкспериментировать: gamma 1.0–2.0)
        gamma = 1.5

        # Приводим к 2D и 1D
        logits_flat = logits.view(-1, logits.size(-1))  # (N, C)
        labels_flat = labels.view(-1)  # (N,)

        # log_softmax и вероятности
        log_probs = nn.functional.log_softmax(logits_flat, dim=-1)  # (N, C)
        probs = log_probs.exp()  # (N, C)

        # Берём p_t и log_p_t для истинного класса
        labels_flat_long = labels_flat.long()
        idx = torch.arange(labels_flat_long.size(0), device=logits.device)
        log_p_t = log_probs[idx, labels_flat_long]  # (N,)
        p_t = probs[idx, labels_flat_long]  # (N,)

        # Веса классов alpha_t (из твоих class_weights)
        class_weights = self.class_weights.to(logits.device)  # (C,)
        alpha_t = class_weights[labels_flat_long]  # (N,)

        # Focal Loss: - alpha_t * (1 - p_t)^gamma * log_p_t
        focal_factor = (1.0 - p_t) ** gamma
        loss = -alpha_t * focal_factor * log_p_t
        loss = loss.mean()
        # --- конец Focal Loss ---

        return (loss, outputs) if return_outputs else loss


trainer = WeightedLossTrainer(
    class_weights=class_weights,
    model=classifier,
    args=training_args,
    train_dataset=labeled_train_set,
    eval_dataset=labeled_validation_set,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print(f"Используется CUDA: {torch.cuda.is_available()}")
print(f"Веса классов: {class_weights}")

trainer.train()

# %%
# Eval на валидации
results = trainer.evaluate()
print("\nРезультаты на валидации:")
for key, value in results.items():
    if isinstance(value, (int, float)):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")

# %%
# Pipeline для inference (берем лучшую модель из trainer)
best_model = trainer.model


def create_classification_pipeline_with_normalization(model, tokenizer, device=-1):
    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    def predict_with_normalization(texts):
        normalized_texts = normalize_batch_texts(texts)
        return clf(normalized_texts, truncation=True, max_length=MAX_LEN)

    return predict_with_normalization


clf_normalized = create_classification_pipeline_with_normalization(
    model=best_model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
)

# %%
# Validation report
validation_texts = list(validation_set["text"])
pred_val = [label2id[x["label"]] for x in clf_normalized(validation_texts)]
true_val = [label2id[val] for val in validation_set["category"]]

print("\n" + "=" * 50)
print("Validation report:")
print("=" * 50)
print(
    classification_report(
        y_true=true_val, y_pred=pred_val, target_names=list_of_categories, digits=4
    )
)

# %%
# Test report
test_texts = list(test_set["text"])
pred_test = [label2id[x["label"]] for x in clf_normalized(test_texts)]
true_test = [label2id[val] for val in test_set["category"]]

print("\n" + "=" * 50)
print("Test report:")
print("=" * 50)
print(
    classification_report(
        y_true=true_test, y_pred=pred_test, target_names=list_of_categories, digits=4
    )
)

# %%


# %%
# Получаем логиты и true labels на валидации (без pipeline)
def get_logits_and_labels(dataset, model, data_collator, batch_size=MINIBATCH_SIZE):
    model.eval()
    device = model.device

    all_logits = []
    all_labels = []

    torch_dataset = dataset.with_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"],
    )

    data_loader = DataLoader(
        torch_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    with torch.no_grad():
        for batch in data_loader:
            # batch содержит input_ids, attention_mask, label
            labels = batch["labels"].to(device)
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            outputs = model(**inputs)
            logits = outputs.logits
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)  # (N, num_classes)
    all_labels = torch.cat(all_labels, dim=0)  # (N,)
    return all_logits, all_labels


# Получаем логиты на валидации
val_logits, val_labels = get_logits_and_labels(
    labeled_validation_set, trainer.model, data_collator
)

# Переводим в numpy
val_probs = F.softmax(val_logits, dim=-1).numpy()
val_true = val_labels.numpy()

ent_idx = label2id["entertainment"]


def predict_with_delta_entertainment(probs, delta):
    """
    probs: numpy array (N, num_classes)
    delta: допускаем, что p(entertainment) >= p(top1) - delta
    """
    preds = []
    for p in probs:
        top1_idx = int(p.argmax())
        top1_prob = float(p[top1_idx])
        ent_prob = float(p[ent_idx])

        # если entertainment и так top1 — берём его
        if top1_idx == ent_idx:
            preds.append(ent_idx)
            continue

        # если entertainment не top1, но его вероятность не сильно ниже top1
        # и достаточно велика сама по себе — перескакиваем в entertainment
        if (top1_prob - ent_prob) <= delta and ent_prob >= 0.20:
            preds.append(ent_idx)
        else:
            preds.append(top1_idx)
    return np.array(preds, dtype=np.int64)


# Подбор лучшего delta по macro F1 на валидации
candidate_deltas = [0.0, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
best_delta = 0.0
best_f1 = -1.0

for d in candidate_deltas:
    preds_d = predict_with_delta_entertainment(val_probs, d)
    f1_macro_d = f1_score(val_true, preds_d, average="macro")
    print(f"delta={d:.3f} => macro F1 (val) = {f1_macro_d:.4f}")
    if f1_macro_d > best_f1:
        best_f1 = f1_macro_d
        best_delta = d

print(f"\nВыбран best_delta={best_delta:.3f} с macro F1 на валидации={best_f1:.4f}")

# %%
# Оценка на валидации и тесте с этим правилом

# Валидация (у нас уже есть val_probs / val_true)
val_preds_threshold = predict_with_delta_entertainment(val_probs, best_delta)
print("\n" + "=" * 50)
print("Validation report (with entertainment threshold tuning):")
print("=" * 50)
print(
    classification_report(
        y_true=val_true,
        y_pred=val_preds_threshold,
        target_names=list_of_categories,
        digits=4,
    )
)

# Тест: сначала собираем логиты, потом применяем то же правило
# Готовим токенизированный тест с label-колонкой
tokenized_test_set = test_set.map(tok, batched=True)
labeled_test_set = tokenized_test_set.add_column(
    "label", [label2id[val] for val in tokenized_test_set["category"]]
)

test_logits, test_labels = get_logits_and_labels(
    labeled_test_set, trainer.model, data_collator
)
test_probs = F.softmax(test_logits, dim=-1).numpy()
test_true = test_labels.numpy()

test_preds_threshold = predict_with_delta_entertainment(test_probs, best_delta)

print("\n" + "=" * 50)
print("Test report (with entertainment threshold tuning):")
print("=" * 50)
print(
    classification_report(
        y_true=test_true,
        y_pred=test_preds_threshold,
        target_names=list_of_categories,
        digits=4,
    )
)
