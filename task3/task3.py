# %%
# Config & imports
import evaluate
import numpy as np
from datasets import load_dataset
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
DATASET_NAME = "Davlan/sib200"
DATASET_LANGUAGE = "rus_Cyrl"
MINIBATCH_SIZE = 32  # smaller per-device batch often generalizes better; gradient_accumulation can be used if needed

# %%
# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# %%
# Load data
train_set = load_dataset(DATASET_NAME, DATASET_LANGUAGE, split="train")
validation_set = load_dataset(DATASET_NAME, DATASET_LANGUAGE, split="validation")
test_set = load_dataset(DATASET_NAME, DATASET_LANGUAGE, split="test")


# %%
# Tokenize
def tok(batch):
    return tokenizer(batch["text"], truncation=True)


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

labeled_train_set = tokenized_train_set.add_column(
    "label", [label2id[val] for val in tokenized_train_set["category"]]
)
labeled_validation_set = tokenized_validation_set.add_column(
    "label", [label2id[val] for val in tokenized_validation_set["category"]]
)

# %%
# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# %%
# Metric
cls_metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return cls_metric.compute(
        predictions=predictions, references=labels, average="macro"
    )


# %%
# Model
classifier = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=n_categories,
    id2label=id2label,
    label2id=label2id,
)

# %%
# Training
training_args = TrainingArguments(
    output_dir="rubert_sib200",
    learning_rate=2e-5,
    per_device_train_batch_size=MINIBATCH_SIZE,
    per_device_eval_batch_size=MINIBATCH_SIZE,
    gradient_accumulation_steps=2,  # effective batch 64
    num_train_epochs=8,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True,
    logging_steps=50,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    seed=SEED,
    data_seed=SEED,
    report_to=["none"],
)

trainer = Trainer(
    model=classifier,
    args=training_args,
    train_dataset=labeled_train_set,
    eval_dataset=labeled_validation_set,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# %%
# Eval
trainer.evaluate()

# %%
# Inference and reports
clf = pipeline("text-classification", model=classifier, tokenizer=tokenizer, device=-1)

pred_val = [x["label"] for x in clf(validation_set["text"])]
true_val = validation_set["category"]
print("Validation report:")
print(classification_report(y_true=true_val, y_pred=pred_val, digits=4))

pred_test = [x["label"] for x in clf(test_set["text"])]
true_test = test_set["category"]
print("Test report:")
print(classification_report(y_true=true_test, y_pred=pred_test, digits=4))
