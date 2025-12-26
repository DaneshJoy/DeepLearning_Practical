# Hugging Face Transformers Fine-tuning

## 0) Setup (once)

```bash
pip install -U transformers datasets accelerate evaluate scikit-learn torch torchvision pillow
# optional (for LoRA/quantization later)
pip install -U peft bitsandbytes
```

------

## 1) Fine-tune a text classifier (Trainer API)

We’ll use **AG News** (4 classes).

```python
ds = load_dataset("ag_news")
ds
```

### A) Tokenize

```python
base_model = "distilbert-base-uncased"
tok = AutoTokenizer.from_pretrained(base_model)

def tokenize(batch):
    return tok(batch["text"], truncation=True)

tokenized = ds.map(tokenize, batched=True)
```

### B) Build model + label maps

```python
num_labels = 4
model = AutoModelForSequenceClassification.from_pretrained(
    base_model, num_labels=num_labels
)
```

### C) Collator + metrics

```python
from transformers import DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score

data_collator = DataCollatorWithPadding(tokenizer=tok)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }
```

### D) Train

```python
args = TrainingArguments(
    output_dir="agnews-distilbert",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(),  # safe on GPU
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tok,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()
```

### E) Use the fine-tuned model

```python
text_clf = pipeline("text-classification", model=trainer.model, tokenizer=tok)
text_clf("The startup raised a new funding round.")
```

*(Optional)* Push to Hub after `huggingface-cli login`:

```python
trainer.push_to_hub("agnews-distilbert-finetuned")
```

------

## 2) Fine-tune an image classifier (Trainer API)

We’ll use **beans** (small, good for classroom).

```python
beans = load_dataset("beans")
# figure out label column safely
label_col = "labels" if "labels" in beans["train"].features else "label"
label_names = beans["train"].features[label_col].names
id2label = {i: n for i, n in enumerate(label_names)}
label2id = {n: i for i, n in enumerate(label_names)}
```

### A) Processor + transforms

```python
base_img = "google/vit-base-patch16-224"
proc = AutoImageProcessor.from_pretrained(base_img)

def transform(example):
    # `image` column is standard in vision datasets
    inputs = proc(images=example["image"], return_tensors="pt")
    out = {k: v.squeeze(0) for k, v in inputs.items()}
    out["labels"] = example[label_col]
    return out

train_t = beans["train"].with_transform(transform)
val_t   = beans["validation"].with_transform(transform)
test_t  = beans["test"].with_transform(transform)
```

### B) Model + collator

```python
imodel = AutoModelForImageClassification.from_pretrained(
    base_img,
    num_labels=len(label_names),
    id2label=id2label,
    label2id=label2id,
)

collator = DefaultDataCollator()
```

### C) Train

```python
args = TrainingArguments(
    output_dir="beans-vit",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=5e-5,
    num_train_epochs=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(),
    remove_unused_columns=False,  # important for image datasets
    report_to="none",
)

def compute_img_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    from sklearn.metrics import accuracy_score
    return {"accuracy": accuracy_score(labels, preds)}

itrainer = Trainer(
    model=imodel,
    args=args,
    train_dataset=train_t,
    eval_dataset=val_t,
    tokenizer=proc,              # lets Trainer save processor with model
    data_collator=collator,
    compute_metrics=compute_img_metrics,
)

itrainer.train()
itrainer.evaluate(test_t)
```

### D) Inference

```python
iclf = pipeline("image-classification", model=itrainer.model, image_processor=proc)
iclf(img, top_k=3)
```

------

## 