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

> **Note:** If you are on a GPU system, always install `pytorch`, `torchvision`, `pytorch-cuda`, and `ultralytics` together.
> This way Conda resolves conflicts. If you install separately, install **pytorch-cuda last** to override the CPU version.

📖 Reference: [Ultralytics Docs](https://docs.ultralytics.com/tasks/)

---

# Model Training Documentary

This repository documents my **training journey step by step**.
I’m writing it in a way that avoids the pitfalls I faced, so others won’t have to “crack their head” while following along.

---

## Step 1 – Organize the Dataset First (Important!)

Before labeling, it’s better to **create the dataset structure and split the images**.
That way, when you use `LabelImg`, you can directly open/save annotations in the correct folders (train/val/test) without having to move them later.

Directory layout:

```
attire_dataset
├───train
│   ├───images
│   └───labels
├───valid
│   ├───images
│   └───labels
└───test
    ├───images
    └───labels
```

* Place \~80% of images into `train/images`.
* Place \~20% into `valid/images`.
* Optionally, keep a small set (\~10%) in `test/images`.
* At this stage, `labels/` folders will be empty — LabelImg will generate them when you annotate.

---

## Step 2 – Label the Images

Now open **LabelImg** and annotate images **inside their respective split folders**:

1. Open `LabelImg` → select **YOLO** format.
2. For **train split**:

   * `Open Dir` → `attire_dataset/train/images`
   * `Save Dir` → `attire_dataset/train/labels`
3. For **valid split**:

   * `Open Dir` → `attire_dataset/valid/images`
   * `Save Dir` → `attire_dataset/valid/labels`
4. Define classes in `classes.txt`:

   ```
   tie
   formal_dress
   kurthi_dupatta
   ```
5. Start drawing bounding boxes and assigning classes.

   * Every image will get a `.txt` with the same name inside the correct `labels/` folder.
   * Empty `.txt` = no objects (valid).

---

## Step 3 – Training the Dataset (First Attempt)

Once labeling is done, you’re ready to train.

Example dataset directory:

```
C:\Users\Amith\OneDrive\Desktop\attire_dataset\
```

Basic training command:

```bash
yolo task=detect mode=train model=yolov8n.pt data="C:/Users/Amith/OneDrive/Desktop/attire_dataset/data.yaml" \
     epochs=80 imgsz=640 batch=8 workers=0 device=cpu name=attire_v1
```

* `train/` and `valid/` are used automatically from `data.yaml`.
* The best weights will be saved to:

  ```
  runs/detect/attire_v1/weights/best.pt
  ```

---

✅ This order (split → label → train) makes the workflow clean and avoids dataset mismatches.

---

Would you like me to now **merge this with the refined process from your other file** (the one with DeepFace integration, remapping script, auto-split helper) so the doc flows like:
**v1 (this process) → v2 (refinements) → v3 (upgrades)**?
