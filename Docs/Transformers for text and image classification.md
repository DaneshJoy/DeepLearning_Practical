# Hugging Face Transformers for **text and image classification**

We’ll start with **pipelines**, then peel back to **Auto-classes** for custom inference + fine-tuning.

I’ve included runnable code you can drop into a notebook/script.

------

## 0) Setup (once)

```bash
pip install -U transformers datasets accelerate evaluate scikit-learn torch torchvision pillow
# optional (for LoRA/quantization later)
pip install -U peft bitsandbytes
```

Basic imports you’ll reuse:

```python
from transformers import (
    pipeline, AutoTokenizer, AutoModelForSequenceClassification,
    AutoImageProcessor, AutoModelForImageClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding, DefaultDataCollator
)
from datasets import load_dataset
import torch, numpy as np
```

------

## 1) Pipelines first (fast wins)

### 1A) Text classification (sentiment)

Concepts: pipeline abstracts tokenization → model → post-processing. Good for demos & quick prototyping.

```python
from transformers import pipeline
clf = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")  # 2-way sentiment
clf(["I absolutely love this!", "This was a waste of time."])
```

Tips:

- `pipeline(..., device=0)` to use GPU.
- Batch by passing a list for speed.

### 1B) Zero-shot text classification

Concepts: NLI models (e.g., BART MNLI) score hypothesis “this text is about ”.

```python
zst = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
zst(
    "We deployed a new model to production and fixed latency issues.",
    candidate_labels=["technology","sports","politics","finance"],
    multi_label=False, # True lets multiple labels be “on”
)
```

### 1C) Image classification

Concepts: Image processors handle resize/normalize; model is usually ViT/ConvNet.

```python
from transformers import pipeline
from PIL import Image
import requests, io

img = Image.open(requests.get("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cat.png", stream=True).raw)
iclf = pipeline("image-classification", model="google/vit-base-patch16-224")
iclf(img, top_k=3)
```

### 1D) Zero-shot **image** classification (CLIP)

Concepts: Joint text–image model compares embeddings.

```python
zsic = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")
zsic(
    img,
    candidate_labels=["tabby cat","dog","car","tiger"],
)
```



------

## 2) From pipelines → Auto-classes (custom inference)

### 2A) Text: raw inference with AutoTokenizer + AutoModel

Concepts: tokenization, logits → softmax, id2label maps.

```python
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

name = "distilbert-base-uncased-finetuned-sst-2-english"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name)
model.eval()

text = "I love teaching transformers!"
enc = tok(text, return_tensors="pt")
with torch.no_grad():
    logits = model(**enc).logits
probs = F.softmax(logits, dim=-1).squeeze().tolist()

id2label = model.config.id2label
{ id2label[i]: float(p) for i,p in enumerate(probs) }
```

### 2B) Image: AutoImageProcessor + AutoModelForImageClassification

Concepts: processor returns pixel_values; same logits→softmax.

```python
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch.nn.functional as F

name = "google/vit-base-patch16-224"
proc = AutoImageProcessor.from_pretrained(name)
imodel = AutoModelForImageClassification.from_pretrained(name)
imodel.eval()

inputs = proc(images=img, return_tensors="pt")
with torch.no_grad():
    logits = imodel(**inputs).logits
probs = F.softmax(logits, dim=-1).squeeze()
top5 = torch.topk(probs, k=5)
{ imodel.config.id2label[int(i)]: float(p) for p,i in zip(top5.values, top5.indices) }
```



------

## 3) Fine-tune a text classifier (Trainer API)

We’ll use **AG News** (4 classes).

```python
from datasets import load_dataset
ds = load_dataset("ag_news")
ds
```

### 3A) Tokenize

```python
base_model = "distilbert-base-uncased"
tok = AutoTokenizer.from_pretrained(base_model)

def tokenize(batch):
    return tok(batch["text"], truncation=True)

tokenized = ds.map(tokenize, batched=True)
```

### 3B) Build model + label maps

```python
num_labels = 4
model = AutoModelForSequenceClassification.from_pretrained(
    base_model, num_labels=num_labels
)
```

### 3C) Collator + metrics

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

### 3D) Train

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

### 3E) Use the fine-tuned model

```python
text_clf = pipeline("text-classification", model=trainer.model, tokenizer=tok)
text_clf("The startup raised a new funding round.")
```

*(Optional)* Push to Hub after `huggingface-cli login`:

```python
trainer.push_to_hub("agnews-distilbert-finetuned")
```

------

## 4) Fine-tune an image classifier (Trainer API)

We’ll use **beans** (small, good for classroom).

```python
beans = load_dataset("beans")
# figure out label column safely
label_col = "labels" if "labels" in beans["train"].features else "label"
label_names = beans["train"].features[label_col].names
id2label = {i: n for i, n in enumerate(label_names)}
label2id = {n: i for i, n in enumerate(label_names)}
```

### 4A) Processor + transforms

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

### 4B) Model + collator

```python
imodel = AutoModelForImageClassification.from_pretrained(
    base_img,
    num_labels=len(label_names),
    id2label=id2label,
    label2id=label2id,
)

collator = DefaultDataCollator()
```

### 4C) Train

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

### 4D) Inference

```python
iclf = pipeline("image-classification", model=itrainer.model, image_processor=proc)
iclf(img, top_k=3)
```

------

## 5) Multi-label text classification (extra)

Key changes:

- Use `problem_type="multi_label_classification"`.
- BCEWithLogitsLoss (handled automatically).
- Sigmoid + thresholding at inference.

Example with a toy setup:

```python
from datasets import load_dataset
ds = load_dataset("go_emotions", "simplified")  # multi-label; labels in "labels"

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
def tok_fn(batch):
    return tok(batch["text"], truncation=True)

tok_ds = ds.map(tok_fn, batched=True)

num_labels = len(tok_ds["train"].features["labels"].feature.names)
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=num_labels,
    problem_type="multi_label_classification",
)

collator = DataCollatorWithPadding(tokenizer=tok)

def compute_multi_metrics(eval_pred, threshold=0.5):
    logits, labels = eval_pred
    probs = 1/(1+np.exp(-logits))
    preds = (probs >= threshold).astype(int)
    from sklearn.metrics import f1_score
    return {"f1_micro": f1_score(labels, preds, average="micro")}

args = TrainingArguments(
    "goemo-mlc",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    evaluation_strategy="epoch",
    fp16=torch.cuda.is_available(),
    report_to="none",
)

trainer = Trainer(
    model=model, args=args,
    train_dataset=tok_ds["train"], eval_dataset=tok_ds["validation"],
    tokenizer=tok, data_collator=collator,
    compute_metrics=compute_multi_metrics
)
trainer.train()
```

Prediction:

```python
def predict_multilabel(texts, threshold=0.5):
    enc = tok(texts, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.sigmoid(logits).cpu().numpy()
    names = tok_ds["train"].features["labels"].feature.names
    out = []
    for row in probs:
        out.append([names[i] for i,p in enumerate(row) if p>=threshold])
    return out
```

------

## 6) Zero-shot tricks you’ll want

- **Text**: use domain labels (even long phrases) and `multi_label=True` to get overlapping categories.
- **Image**: CLIP with prompts (e.g., `["a photo of a labrador", "a photo of a beagle"]`) improves accuracy.

------

## 7) Practical extras (short, powerful)

**a) Class imbalance (text or image)**
 Use weighted loss:

```python
# Example for text: pass class weights via Trainer
from torch.nn.functional import cross_entropy
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute weights from your data beforehand
        weights = torch.tensor([0.3, 0.7], device=logits.device)  # example for 2 classes
        loss = cross_entropy(logits, labels, weight=weights)
        return (loss, outputs) if return_outputs else loss
```

**b) Speed & memory**

- `fp16=True` or `bf16=True` (on supported GPUs).
- Gradient accumulation for big batches: `gradient_accumulation_steps`.
- `bitsandbytes` 8-bit loading:

```python
from transformers import BitsAndBytesConfig
bnb = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=num_labels, quantization_config=bnb, device_map="auto")
```

**c) LoRA fine-tuning (PEFT)**

```python
from peft import LoraConfig, get_peft_model
lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["q_lin","v_lin","query","value"])  # target names vary by arch
peft_model = get_peft_model(model, lora)
```

**d) Confusion matrix (after eval)**

```python
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
preds = trainer.predict(tokenized["test"]).predictions.argmax(-1)
cm = confusion_matrix(tokenized["test"]["label"], preds)
print(cm)
```

**e) Save & load**

```python
trainer.save_model("checkpoint-best")
# later:
tok = AutoTokenizer.from_pretrained("checkpoint-best")
model = AutoModelForSequenceClassification.from_pretrained("checkpoint-best")
```

**f) Simple deployment (FastAPI + pipeline)**

```python
# pip install fastapi uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI()

pipe = pipeline("text-classification", model="checkpoint-best", tokenizer="checkpoint-best")

class Item(BaseModel):
    text: str

@app.post("/predict")
def predict(item: Item):
    return pipe(item.text)

# run: uvicorn app:app --reload --port 8000
```

------

## 8) Common gotchas

- For vision datasets with `with_transform`, set `remove_unused_columns=False` in `TrainingArguments`.
- Labels column name varies (`label` vs `labels`); detect dynamically as shown.
- Multi-label needs `problem_type="multi_label_classification"` or a custom loss.
- Large models? Try `bitsandbytes` 8-bit + LoRA; don’t push full-precision checkpoints if you trained LoRA adapters—save adapters separately (`peft_model.save_pretrained`).
