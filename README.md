# Model-Training
This Repo is about my model training process and the later upgradation undergone . this is a private repo for my documentary purpose

# Collecting the required data
First we need to collect the required data, in our case it was to collect images like duppata - kurti , tucked shirts - untucked shirts , tie
Gather positives for each class (aim for at least ~150–300 images/class to start).
- Also collect **negatives** (people without ties, casual wear, random scenes). Negatives help reduce false positives.
- Mix lighting, angles, backgrounds. Prefer JPG/PNG.
we used a 3rd party extension for downloading the images named **fatkun** it help in downloading bulk images.

# labeling the images
after collecting the images we need to label them (box labels) for object detection, we used the **LabelImg** tool for annotating the images.
1. `labelImg` → choose **YOLO** format.
2. Set “Save Dir” to your `labels` folder and “Open Dir” to your images folder.
3. Create a **classes list**: in LabelImg, press `View → YOLO` and set your classes list (or ensure a `classes.txt` exists with):
    ```
    tie
    formal_dress
    kurthi_dupatta
    ```
4. Draw boxes. Make sure you select the correct class before each box.
5. Each image must have a `.txt` with the **same filename**. (Empty `.txt` means no objects = okay.)

# training the dataset (first time training)
Make a clean dataset directory, e.g. `C:\Users\Amith\OneDrive\Desktop\attire_dataset\`:

```
attire_dataset/
  images/
    train/
    val/
    test/           (optional at first)
  labels/
    train/
    val/
    test/
```

Split roughly **80/20** train/val (and optionally 10% test taken from train). Keep filenames matched across `images/` and `labels/`.
