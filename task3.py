# %%
# Config & imports
import collections
import re

import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import classification_report
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    pipeline,
)

SEED = 42
MODEL_NAME = "DeepPavlov/rubert-base-cased"
# MODEL_NAME = "Den4ikAI/rubert-large-squad"
DATASET_NAME = "Davlan/sib200"
DATASET_LANGUAGE = "rus_Cyrl"
MINIBATCH_SIZE = 32


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


# %%
# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
MAX_LEN = 256

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


# %%
# Вычисление весов классов
def compute_class_weights(labels, label2id, method="inverse"):
    """
    Вычисление весов классов

    Args:
        labels: список меток
        label2id: словарь соответствия меток индексам
        method: метод вычисления весов
            - 'balanced': обратная частота (sklearn style)
            - 'inverse': обратные частоты
            - 'sqrt': квадратные корни обратных частот
    """
    # Подсчет частот классов
    class_counts = collections.Counter(labels)
    n_classes = len(class_counts)

    # Преобразование меток в индексы
    label_indices = [label2id[label] for label in labels]

    # Подсчет количества примеров по индексам
    class_counts_by_idx = [0] * n_classes
    for idx in label_indices:
        class_counts_by_idx[idx] += 1

    # Вычисление весов
    if method == "balanced":
        # Метод из sklearn (обратная частота)
        total = sum(class_counts_by_idx)
        weights = [
            total / (n_classes * count) if count > 0 else 1.0
            for count in class_counts_by_idx
        ]

    elif method == "inverse":
        # Простые обратные частоты
        max_count = max(class_counts_by_idx)
        weights = [
            max_count / count if count > 0 else 1.0 for count in class_counts_by_idx
        ]

    elif method == "sqrt":
        # Квадратные корни обратных частот
        max_count = max(class_counts_by_idx)
        weights = [
            np.sqrt(max_count / count) if count > 0 else 1.0
            for count in class_counts_by_idx
        ]

    else:
        raise ValueError(f"Неизвестный метод: {method}")

    # Нормализация весов
    weights = [w / sum(weights) * n_classes for w in weights]

    print(f"\nВеса классов (метод: {method}):")
    for idx, (label, count) in enumerate(class_counts.most_common()):
        print(f"  {label} (idx={idx}): count={count}, weight={weights[idx]:.4f}")

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
class_weights = compute_class_weights(
    train_set["category"], label2id, method="balanced"
)

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
)

# %%
# Training с учетом весов классов - УПРОЩЕННАЯ ВЕРСИЯ без кастомного Trainer
# Мы будем использовать стандартный Trainer, но с weighted loss

training_args = TrainingArguments(
    output_dir="rubert_sib200_weighted",
    learning_rate=5e-5,
    per_device_train_batch_size=MINIBATCH_SIZE,
    per_device_eval_batch_size=MINIBATCH_SIZE,
    gradient_accumulation_steps=6,
    num_train_epochs=20,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",  # Используем eval_f1, а не eval_macro_f1
    greater_is_better=True,
    logging_steps=50,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    seed=SEED,
    data_seed=SEED,
    report_to=["none"],
    fp16=torch.cuda.is_available(),
    no_cuda=not torch.cuda.is_available(),
    label_smoothing_factor=0.1,
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
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
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
