# Hugging Face Transformers for **object detection**

- with both the **pipeline** approach and the **Auto-classes** way, plus a tiny helper to **visualize bounding boxes**.

## 0) Setup (once)

```bash
pip install -U transformers pillow torch torchvision pillow
```
------

## 1) Quick way: pipelines → object detection (+ visualization)

```python
from transformers import pipeline
from PIL import Image, ImageDraw, ImageFont

# 1) Load an image (local file path shown; URL works too if you stream with requests)
img = Image.open("example.jpg").convert("RGB")  # replace with your image path

# 2) Create pipeline (DETR is a great default)
detector = pipeline(
    task="object-detection",
    model="facebook/detr-resnet-50",  # or "facebook/detr-resnet-101"
    device=0 if torch.cuda.is_available() else -1,
)

# 3) Run detection
preds = detector(img)  # list of dicts: {'score','label','box':{'xmin','ymin','xmax','ymax'}}

# 4) Visualize
def draw_detections(image, predictions, thresh=0.7, max_labels=50):
    im = image.copy()
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default()
    except:
        font = None

    shown = 0
    for p in predictions:
        if p["score"] < thresh: 
            continue
        box = p["box"]
        xyxy = (box["xmin"], box["ymin"], box["xmax"], box["ymax"])
        draw.rectangle(xyxy, outline="red", width=3)
        text = f'{p["label"]}: {p["score"]:.2f}'
        # background for text
        tw, th = draw.textsize(text, font=font)
        x0, y0 = box["xmin"], max(0, box["ymin"] - th - 2)
        draw.rectangle([x0, y0, x0 + tw + 4, y0 + th + 2], fill="red")
        draw.text((x0 + 2, y0 + 1), text, fill="white", font=font)
        shown += 1
        if shown >= max_labels:
            break
    return im

vis = draw_detections(img, preds, thresh=0.7)
vis.show()  # or vis.save("detections_pipeline.jpg")
```

**Notes**

- Move the threshold slider live (e.g., 0.3 vs 0.9) to show precision/recall trade-off.
- Swap models: try `"facebook/detr-resnet-101"` or `"yolos-small"` for variety.

------

## 2) Under the hood: AutoImageProcessor + AutoModelForObjectDetection

```python
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image, ImageDraw, ImageFont

name = "facebook/detr-resnet-50"
processor = AutoImageProcessor.from_pretrained(name)
model = AutoModelForObjectDetection.from_pretrained(name)
model.eval()

img = Image.open("example.jpg").convert("RGB")

# 1) Preprocess
inputs = processor(images=img, return_tensors="pt")

# 2) Forward
with torch.no_grad():
    outputs = model(**inputs)

# 3) Post-process to image scale
# target_sizes expects (height, width)
target_sizes = torch.tensor([img.size[::-1]])
results = processor.post_process_object_detection(
    outputs, threshold=0.7, target_sizes=target_sizes
)[0]
# results keys: 'scores', 'labels' (ids), 'boxes' (xyxy)

# 4) Visualize (reusing the same helper but adapted to Auto outputs)
id2label = model.config.id2label

def draw_auto_detections(image, scores, labels, boxes, max_labels=50):
    im = image.copy()
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default()
    except:
        font = None
    shown = 0
    for score, label_id, box in zip(scores, labels, boxes):
        lbl = id2label[int(label_id)]
        text = f"{lbl}: {float(score):.2f}"
        x0, y0, x1, y1 = box.tolist()
        draw.rectangle((x0, y0, x1, y1), outline="red", width=3)
        tw, th = draw.textsize(text, font=font)
        y_text = max(0, y0 - th - 2)
        draw.rectangle([x0, y_text, x0 + tw + 4, y_text + th + 2], fill="red")
        draw.text((x0 + 2, y_text + 1), text, fill="white", font=font)
        shown += 1
        if shown >= max_labels:
            break
    return im

vis2 = draw_auto_detections(img, results["scores"], results["labels"], results["boxes"])
vis2.show()  # or vis2.save("detections_auto.jpg")
```

**Notes**

- The processor handles resizing/normalizing and **post-processing** to real-image coordinates.
- `id2label` translates numeric class IDs to names.

------

## 3) (Optional) Batched inference + NMS

DETR already has built-in set prediction (no NMS by design). If you try YOLO-style models (e.g., YOLOS), you might want NMS. Here’s a minimal example with torchvision NMS:

```python
import torchvision
def nms_filter(boxes, scores, iou_thr=0.5, top_k=None):
    keep = torchvision.ops.nms(boxes, scores, iou_thr)
    if top_k is not None:
        keep = keep[:top_k]
    return keep

# Example usage on Auto results (tensors):
keep_idx = nms_filter(results["boxes"], results["scores"], iou_thr=0.5)
scores = results["scores"][keep_idx]
labels = results["labels"][keep_idx]
boxes  = results["boxes"][keep_idx]
vis_nms = draw_auto_detections(img, scores, labels, boxes)
vis_nms.show()
```

------

## 4) Zero-shot **open-vocabulary** detection

This is an advanced, evolving area. If you want a “wow” moment, demo grounding with a model like **OWL-ViT**:

```python
from transformers import pipeline
detector = pipeline("zero-shot-object-detection", model="google/owlvit-base-patch32")
preds = detector(
    img,
    candidate_labels=["cat", "laptop", "cup", "dog", "keyboard"],
)
# preds format matches object-detection pipeline (score, label, box)
vis = draw_detections(img, preds, thresh=0.2)
vis.show()
```

------

## 5) Minimal fine-tuning pointers

Full detection training is heavier than classification. If you want to introduce it later:

- Start with **DETR** on a small subset of COCO or a tiny custom dataset (Pascal VOC format → COCO, or write a `datasets` loader).
- Use `AutoModelForObjectDetection` + `Trainer`, but expect to write a custom `collate_fn` and format `labels` as bounding boxes + class ids per image.
- For limited time, prefer **inference + annotation review** unless you can dedicate a long time to training.

