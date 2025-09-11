Perfect 👍 I’ll **improvise** your draft into a more polished, professional, and documentary-style write-up while keeping your tone (personal + step recording).

Here’s the improved version:

---

# Setting up the Environment for Model Training & Testing

I chose to set up the training environment using an **Anaconda virtual environment** instead of the system’s Python environment.
The main reason: training requires **Python 3.8 or lower**, while most systems usually run higher versions (≥3.9), which may cause compatibility issues.
To avoid this, I created a dedicated Anaconda environment with **Python 3.8**.

### Installing Required Packages

* **Option 1: Install Ultralytics only**

```bash
conda install -c conda-forge ultralytics
```

* **Option 2: Install all required packages together (recommended for NVIDIA GPUs with CUDA)**

```bash
conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics
```

> **Note:**
> If you are setting up in a CUDA-enabled environment, it is best practice to install **Ultralytics, PyTorch, and PyTorch-CUDA together in one command**.
> This lets Conda resolve dependencies automatically. If you install them separately, make sure to install **pytorch-cuda last** to override the default CPU-only PyTorch package.

📖 Reference: [Ultralytics Documentation](https://docs.ultralytics.com/tasks/)

---

# Model Training Documentary

This repository documents my entire **model training process**, along with the **upgrades and improvements** I applied over time.
It serves as my personal logbook.

---

## Step 1 – Collecting Data

For this project, I needed to detect clothing items such as:

* **Dupatta with Kurti**
* **Tucked shirts vs Untucked shirts**
* **Tie**

### Data Collection Strategy:

* Collected **positive samples** for each class (\~150–300 images per class to begin with).
* Collected **negative samples** (people without ties, casual wear, random backgrounds). Negatives are crucial to reduce false positives.
* Ensured **diversity** in lighting conditions, angles, and backgrounds.
* Used a Chrome extension **Fatkun** for bulk image downloading. This significantly sped up dataset creation.

---

## Step 2 – Labeling the Images

Annotation was done using **LabelImg** in YOLO format.

1. Open `labelImg` → choose **YOLO** format.
2. Set **Save Dir** = `labels/` folder, **Open Dir** = `images/` folder.
3. Define the classes (either in `View → YOLO` or via a `classes.txt` file):

   ```
   tie
   formal_dress
   kurthi_dupatta
   ```
4. Draw bounding boxes carefully and select the correct class for each.
5. Every image must have a matching `.txt` annotation file.

   * Example: `image1.jpg` → `image1.txt`.
   * Empty `.txt` is valid (means no objects).

---

## Step 3 – Preparing the Dataset Directory

Created a clean dataset directory at:

```
C:\Users\Amith\OneDrive\Desktop\attire_dataset\
```

Folder structure:

```
attire_dataset/
  images/
    train/
    val/
    test/    (optional at first)
  labels/
    train/
    val/
    test/
```

* Train/Val split: **80/20** ratio.
* Optionally, I set aside \~10% of the training set for testing.
* Important: Filenames of images and labels must match exactly across the folders.

---
Would you like me to **extend this further into a versioned timeline style** (like `v1 First Training`, `v2 Refined Training`, `v3 GPU Training`) — so your repo looks like a research documentary with clear upgrades?
