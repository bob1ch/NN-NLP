# YOLOv8 Object Detection Project

This project uses **Ultralytics YOLOv8** to train a custom object detection model on a card dataset. The pipeline covers installation, training, validation, and testing with performance evaluation.

---

## 📦 Installation

Install the required package:

```bash
pip install ultralytics
```

Dependencies (automatically installed with `ultralytics`):

* Python ≥ 3.12
* PyTorch ≥ 2.4.1
* OpenCV
* NumPy
* Matplotlib
* Pillow
* Pandas
* Seaborn
* SciPy
* Requests
* Torchvision

> Note: CUDA-enabled GPU is recommended for faster training.

---

## 🗂 Dataset

The dataset is structured for YOLO:

```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

The `data.yaml` file specifies the dataset path and class labels.

---

## 🚀 Training

Train the YOLOv8 model using:

```bash
yolo detect train \
  data=data.yaml \
  model=yolov8n.pt \
  epochs=15 \
  patience=3 \
  seed=4578 \
  cache=True \
  name=first \
  close_mosaic=5 \
  plots=True
```

**Key options explained:**

* `model=yolov8n.pt` – Pretrained YOLOv8 nano model.
* `epochs=15` – Number of training epochs.
* `patience=3` – Early stopping patience.
* `cache=True` – Cache dataset in RAM for faster training.
* `close_mosaic=5` – Training augmentation parameter.
* `plots=True` – Generate training plots.

Training example results:

| Epoch | Box Loss | Class Loss | DFL Loss | mAP50 |
| ----- | -------- | ---------- | -------- | ----- |
| 1     | 1.153    | 4.26       | 0.9812   | 0.074 |
| 10    | 0.4214   | 0.6296     | 0.8075   | 0.756 |
| 15    | 0.4695   | 0.6607     | 0.8124   | 0.876 |

---

## ✅ Validation

Validate the trained model:

```bash
yolo val \
  model=runs/detect/first/weights/best.pt \
  data=data.yaml \
  plots=True \
  split=val \
  name=val1
```

Example metrics on the validation set:

* **mAP50:** 0.879
* **Precision:** 0.90
* **Recall:** 0.941

---

## 🧪 Testing

Test on the held-out dataset:

```bash
yolo val \
  model=runs/detect/first/weights/best.pt \
  data=data.yaml \
  plots=True \
  split=test \
  name=test1
```

Example metrics on the test set:

* **mAP50:** 0.881
* **Precision:** 0.911
* **Recall:** 0.932

---

## 📊 Results

* Model size: ~6.3 MB (best weights)
* Total parameters: 3,015,788
* GPU used: NVIDIA GeForce RTX 3050 (7.8 GB VRAM)
* Training time: ~0.63 hours for 15 epochs

All results are saved in the `runs/detect/` folder with plots, weights, and logs.

---

## 🔗 References

* [YOLOv8 Documentation](https://docs.ultralytics.com/modes/train)
* [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
