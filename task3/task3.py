# %%
# Config & imports
import collections
import re

import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset, concatenate_datasets, load_dataset
from sklearn.metrics import classification_report
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
# MODEL_NAME = "Den4ikAI/rubert-large-squad"
# MODEL_NAME = "FacebookAI/xlm-roberta-large"
DATASET_NAME = "Davlan/sib200"
DATASET_LANGUAGE = "rus_Cyrl"
MINIBATCH_SIZE = 16


# %%
# Функции для нормализации текста
def normalize_text(text):
    """
    Щадящая нормализация: Unicode NFKC + тримминг пробелов без смены регистра
    и удаления пунктуации (важно для cased-моделей).
    """
    if not isinstance(text, str):
        return ""
    import unicodedata

    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_batch_texts(texts):
    """
    Нормализация батча текстов
    """
    return [normalize_text(text) for text in texts]


# === Аугментации текста ===


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
    text = aug_sentence_shuffle(text, p=0.35)
    text = aug_punct_swap(text, p=0.25)
    text = aug_truncate_middle(text, p=0.25)
    return text


# %%
# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
MAX_LEN = 512

# %%
# Load data
train_set = load_dataset(DATASET_NAME, DATASET_LANGUAGE, split="train")
validation_set = load_dataset(DATASET_NAME, DATASET_LANGUAGE, split="validation")
test_set = load_dataset(DATASET_NAME, DATASET_LANGUAGE, split="test")

# %%
# Применяем нормализацию ко всем наборам данных
print("Применяем нормализацию текстов...")


# Функция для нормализации датасета
def normalize_dataset(dataset):
    normalized_texts = normalize_batch_texts(dataset["text"])
    return dataset.remove_columns(["text"]).add_column("text", normalized_texts)


# Нормализуем все наборы данных
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

# Точечное балансирующее аугментирование: усиливаем редкие классы
# По распределению: entertainment и geography самые редкие
counts = collections.Counter(train_set["category"])
rare_classes = {"entertainment", "geography"}
if len(counts) > 0:
    # целимся подтянуть редкие классы до уровня самого частого класса
    target = max(counts.values()) - 30
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
        weights = [1.0] * n_classes
        for i, lbl in enumerate(classes_in_order):
            if lbl in {"entertainment", "geography"}:
                weights[i] = 2.0
    else:
        raise ValueError(f"Unknown method: {method}")

    print("\nВеса классов:")
    for i, lbl in enumerate(classes_in_order):
        print(
            f"  idx={i}, label={lbl}, count={class_counts_by_idx[i]}, weight={weights[i]:.4f}"
        )

    return torch.tensor(weights, dtype=torch.float32)


# %%
# Tokenize с нормализованными текстами
def tok(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)


tokenized_train_set = train_set.map(tok, batched=True)
tokenized_validation_set = validation_set.map(tok, batched=True)

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

# Вычисляем веса классов ДО создания labeled наборов
print("\nВычисление весов классов для тренировочного набора...")
class_weights = compute_class_weights(train_set["category"], label2id, method="sqrt")

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
# Metric - вернемся к простой версии для совместимости
cls_metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    # Вычисляем только macro F1 для совместимости
    f1_macro = cls_metric.compute(
        predictions=predictions, references=labels, average="macro"
    )["f1"]

    # Вычисляем accuracy
    accuracy = (predictions == labels).mean()

    return {
        "f1": f1_macro,  # Переименуем обратно в f1 для совместимости
        "accuracy": accuracy,
    }


# %%
# Model
classifier = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=n_categories,
    id2label=id2label,
    label2id=label2id,
    classifier_dropout=0.3,
    hidden_dropout_prob=0.3,
    attention_probs_dropout_prob=0.3,
)

# %%

training_args = TrainingArguments(
    output_dir="rubert_sib200_weighted",
    learning_rate=2e-5,
    per_device_train_batch_size=MINIBATCH_SIZE,
    per_device_eval_batch_size=MINIBATCH_SIZE,
    gradient_accumulation_steps=2,
    num_train_epochs=12,
    weight_decay=0.05,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_steps=50,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    seed=SEED,
    data_seed=SEED,
    report_to=["none"],
    fp16=torch.cuda.is_available(),
    no_cuda=not torch.cuda.is_available(),
    label_smoothing_factor=0.05,
    gradient_checkpointing=True,
)


# Кастомный Trainer с взвешенным лоссом и EarlyStopping
class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
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
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

print(f"Используется CUDA: {torch.cuda.is_available()}")
print(f"Веса классов: {class_weights}")

trainer.train()

# %%
# Eval
results = trainer.evaluate()
print("\nРезультаты на валидации:")
for key, value in results.items():
    print(f"  {key}: {value:.4f}")


# %%
# Создаем пайплайн с нормализацией
def create_classification_pipeline_with_normalization(model, tokenizer, device=-1):
    """
    Создает пайплайн для классификации с автоматической нормализацией текста
    """
    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    def predict_with_normalization(texts):
        # Нормализуем тексты перед классификацией
        normalized_texts = normalize_batch_texts(texts)
        return clf(normalized_texts, truncation=True, max_length=MAX_LEN)

    return predict_with_normalization


# Используем пайплайн с нормализацией
clf_normalized = create_classification_pipeline_with_normalization(
    model=classifier,
    tokenizer=tokenizer,
    device=-1,
)

# Валидационный набор
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

# Тестовый набор
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
