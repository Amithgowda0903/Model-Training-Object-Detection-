# Setting up the Environment for Model Training & Testing

I used an **Anaconda virtual environment** instead of the system environment.
Reason: training requires **Python 3.8 or lower**, while most systems come with higher versions (≥3.9).
So, I created a Conda venv with Python 3.8 for compatibility.

### Installing Required Packages

* **Basic install (Ultralytics only)**

```bash
conda install -c conda-forge ultralytics
```

* **Recommended install (with CUDA for NVIDIA GPUs)**

```bash
conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics
```

> **Note:**
> If you are on a GPU system, always install `pytorch`, `torchvision`, `pytorch-cuda`, and `ultralytics` together.
> This way Conda resolves conflicts. If you install separately, install **pytorch-cuda last** to override the CPU version.

📖 Reference: [Ultralytics Docs](https://docs.ultralytics.com/tasks/)

---

# Dataset Preparation & Labeling

## Step 1 – Organize the Dataset First

Before labeling, create the dataset directory structure and split the images.

```
attire_dataset
├───train
│   ├───images
│   └───labels
├───valid
│   ├───images
│   └───labels
└───test
|   ├───images
|   └───labels
|__data.yaml
```

* Place \~80% images in `train/images`.
* Place \~20% in `valid/images`.
* Optionally, keep 10% in `test/images`.
* Labels folder will be filled during annotation.

---

## Step 2 – Label the Images (Object Detection)

At this stage, we are doing **object detection labeling**.
This means we annotate bounding boxes around specific classes (tie, kurti, etc.) so that YOLO can detect and classify them.

Later, I switched to **instance segmentation** for more fine-grained results, but here we stick with detection.

Use **[LabelImg](https://sourceforge.net/projects/labelimg.mirror/)**:

1. Select **YOLO format**.
2. For **train split**:

   * `Open Dir` → `attire_dataset/train/images`
   * `Save Dir` → `attire_dataset/train/labels`
3. For **valid split**:

   * `Open Dir` → `attire_dataset/valid/images`
   * `Save Dir` → `attire_dataset/valid/labels`
4. Define your classes in `classes.txt`:

   ```
   tie
   formal_dress
   kurthi_dupatta
   ```
5. Annotate carefully. Each image will generate a `.txt` file with the same name under `labels/`.

---



## Step 2.5 – Creating the `data.yaml` File

After labeling the images with **LabelImg**, we need to tell YOLO **where the dataset is located** and **which classes we are detecting**.
This is done with a file named `data.yaml`.so we need to copy and save a yaml file with the below info modify the names as per yor dataset.
you can copy the below example yaml to a text file and save it as yaml.

###  Example: `data.yaml`

```yaml
# Path to the root dataset folder
path: C:/Users/Amith/OneDrive/Desktop/attire_dataset

# Train/Val/Test directories (relative to path above)
train: train/images
val: valid/images
test: test/images   # optional

# Number of classes
nc: 3

# Class names (order matters)
names: [tie, formal_dress, kurthi_dupatta]
```

* The `path:` field points to your **dataset root folder**.
* `train`, `val`, `test` are **relative paths** inside that dataset folder.
* `nc` = number of classes (here = 3).
* `names` = list of class names in the **same order** as your `classes.txt` used in LabelImg.

> Important: If you don’t create this `data.yaml`, YOLO won’t know what classes exist or where your dataset is stored.

---

# Step 3 – Training the Dataset

## Model Path Requirement

The YOLO pretrained model file (e.g. `yolov8n.pt`) must be **either**:

* Present inside the Conda environment’s `envs` folder, **or**
* You must provide the **absolute path** to the file in the training command.

---

## Training Commands

### CPU Training Example

```bash
yolo task=detect mode=train model=yolov8n.pt data="C:/Users/name/OneDrive/Desktop/attire_dataset/data.yaml" \
     epochs=80 imgsz=640 batch=8 workers=0 device=cpu name=attire_v1
```

### GPU Training Example

In my case I trained on an **NVIDIA GPU**, so I set the device as `0`:

```bash
yolo task=detect mode=train model=yolov8n.pt data="C:/Users/name/OneDrive/Desktop/attire_dataset/data.yaml" \
     epochs=80 imgsz=640 batch=8 workers=2 device=0 name=attire_v1
```

---

## Hyperparameters Explained

* **epochs** → number of complete passes through the dataset (higher = longer training, better results up to a point).
* **imgsz** → image size used for training (default 640px; higher = more detail but slower).
* **batch** → number of images processed per step (limited by GPU/CPU memory).
* **workers** → number of CPU threads for data loading (0 = safe for Windows, >0 speeds up on Linux).
* **device** → where to train (`cpu`, `0` for GPU:0, `0,1` for multiple GPUs).
* **name** → experiment name (creates `runs/detect/{name}/`).

---

## Training Outputs – `best.pt` vs `last.pt`

After training, you’ll get two key weight files inside `runs/detect/attire_v1/weights/`:

* **best.pt** → model checkpoint with the **highest validation accuracy (mAP)** during training.
* **last.pt** → model checkpoint from the **final training epoch**, regardless of performance.

> In practice, you usually deploy **best.pt**, but sometimes `last.pt` is useful if you want to continue training later.

---

## Monitoring Training

YOLO automatically generates **graphs and logs** during training.

* Monitor **losses** (box loss, cls loss, dfl loss).
* Track **mAP, precision, recall** for train/val.
* Learning how to read these curves is crucial for diagnosing overfitting, underfitting, or data imbalance.

---

## Validation & Testing

* **Validation** → evaluates the model on your `valid/` dataset. This checks accuracy on known but unseen data.
* **Testing** → evaluates the model on a separate `test/` dataset (completely unseen during training & validation).

### Validate Command

```bash
yolo task=detect mode=val model="runs/detect/attire_v1/weights/best.pt" data="C:/Users/name/OneDrive/Desktop/attire_dataset/data.yaml"
```

### Test Command

```bash
yolo task=detect mode=predict model="runs/detect/attire_v1/weights/best.pt" source="C:/Users/name/OneDrive/Desktop/attire_dataset/test/images"
```

---

# Step 4 – PyCharm Setup

1. Install **PyCharm Community Edition**.
2. Create a new project and set the interpreter to your **Conda environment**:

   ```
   anaconda3/envs/<your_env_name>
   ```
3. Install additional dependencies in the same environment:

   ```bash
   pip install deepface tensorflow opencv-python matplotlib pandas
   ```

This ensures everything runs inside the Conda venv linked to PyCharm.

---

# Step 5 – Project Structure

My project folder looked like this after setup:

```
C:\Users\name\OneDrive\Desktop\Attendance System
│   best.pt
│   trial.py
│   yolov8n-face-lindevs.pt
│
├───Database
     └───name
            image_of_person1.jpg
            Image_of_person1_2.png
            Image_of_person1_3.jpg
```

---

# Step 6 – Code Execution

Here’s the execution script combining **YOLO (attire detection)** + **DeepFace (face recognition)**:

```python
import cv2
from ultralytics import YOLO
from deepface import DeepFace

# 1) Face detection model (YOLO pretrained for faces)
face_model = YOLO(r"C:\Users\name\OneDrive\Desktop\Attendance System\yolov8n-face-lindevs.pt")

# 2) Attire detection model (your trained model)
attire_model = YOLO(r"C:\Users\name\OneDrive\Desktop\Attendance System\runs\detect\attire_v1\weights\best.pt")

# DeepFace database path
db_path = r"C:\Users\name\OneDrive\Desktop\Attendance System\Database"

# Classes to display
SHOW_CLASSES = {"tie", "formal_dress", "kurthi_dupatta"}

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Cannot access camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Face Recognition ---
    face_results = face_model(frame, conf=0.5, verbose=False)
    for r in face_results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            face_roi = frame[y1:y2, x1:x2]
            label = "Unknown"

            try:
                dfs = DeepFace.find(img_path=face_roi, db_path=db_path, enforce_detection=False, silent=True)
                if len(dfs) > 0 and not dfs[0].empty:
                    identity_path = dfs[0].iloc[0]['identity']
                    person = identity_path.split("\\")[-2].split("/")[-2]
                    label = person
            except Exception:
                pass

            cv2.rectangle(frame, (x1, y1), (x2, y2), (36, 255, 12), 2)
            cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36, 255, 12), 2)

    # --- Attire Detection ---
    attire_results = attire_model(frame, conf=0.35, iou=0.5, verbose=False)
    for r in attire_results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = attire_model.names[cls_id]
            if cls_name not in SHOW_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
            cv2.putText(frame, cls_name, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)

    cv2.imshow("Attendance + Attire Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```
---

## How YOLO and DeepFace Work Together

In this project, we actually use **two separate models**:

1. **YOLO for Object Detection**
   a. [yolov8n.pt](https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8n.pt)
   b. [yolov8n-face-lindevs.pt](https://github.com/lindevs/yolov8-face/releases/download/1.0.1/yolov8n-face-lindevs.pt)

   * One YOLO model is trained specifically for **faces** (`yolov8n-face-lindevs.pt`).
   * Another YOLO model is trained for **attire detection** (`best.pt`, generated from our training).
   * YOLO’s job is to **detect objects in the frame** (faces or clothing items) by drawing bounding boxes and classifying them.

3. **DeepFace for Face Recognition**

   * Once YOLO detects the **face region**, we crop that part of the frame and pass it to DeepFace.
   * DeepFace compares the cropped face with images in the **Database/** folder.
   * It then recognizes the person (e.g., identifying “Amith” if it matches the stored image).

So, YOLO = *“Where is it?”* → **Detection**
DeepFace = *“Who is it?”* → **Recognition**

---

## Why Download [YOLOv8n](https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8n.pt) Pretrained Models If We Already Have `best.pt`?

This is a common doubt:

* `best.pt` is the **custom-trained weights** for our attire detection task.
* But to **train this model in the first place**, YOLO needs a **base pretrained model** (`yolov8n.pt`).
* During training, YOLO uses transfer learning — starting from `yolov8n.pt` and gradually fine-tuning it on our attire dataset.
* After training completes, YOLO saves:

  * `best.pt` → checkpoint with the best validation score
  * `last.pt` → final epoch checkpoint

So, the `yolov8n.pt` we downloaded wasn’t wasted — it was the **foundation** for creating our `best.pt`.
Without it, our model would have to learn everything from scratch, which would require **much more data and compute**.

---

## 🔚 Conclusion Note

While this process (object detection with YOLOv8) worked well for **basic recognition** of ties, kurtis, and formal wear, it **failed to consistently perform** for my **attire detection task**.

The main issue was that bounding boxes alone were **not enough** to clearly distinguish overlapping or partially visible clothing items (for example, **kurti with dupatta vs. formal dress**). The model often confused classes in cases where boundaries were not clear, lighting varied, or clothing overlapped.

This made me realize that **object detection has its limits** for attire-based classification, especially in real-world scenarios like an **attendance system**, where precision matters.

-> For better accuracy, I later upgraded to **instance segmentation** (YOLO segmentation models), which doesn’t just draw boxes but also **outlines the exact shape of the clothing**. This helped the model learn finer details and reduced misclassifications.

### If you are following this documentation:

* Treat this section as the **baseline training (v1)**.
* Study the **next stage of training (v2: segmentation upgrade)** where I explain how and why I switched to instance segmentation, and the improvements I observed.

