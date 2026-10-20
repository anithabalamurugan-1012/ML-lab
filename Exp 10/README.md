# Retinal Disease Screening Lab

An educational, interactive single-page web application demonstrating the end-to-end machine learning workflow for Diabetic Retinopathy (DR) grading from retinal fundus photographs.

---

## ⚠️ Important Educational and Clinical Disclaimer

> **FOR EDUCATIONAL AND ACADEMIC DEMONSTRATION USE ONLY**  
> This software is an educational prototype developed to illustrate image preprocessing, computer vision feature extraction, and machine learning classification. It is **not clinically validated, certified, or approved by the FDA, CE, or any healthcare regulatory authority**.
>
> - **DO NOT USE FOR MEDICAL DIAGNOSIS OR CLINICAL DECISION-MAKING.**
> - **DO NOT USE TO DIRECT OR ALTER PATIENT TREATMENT.**
> - Model scores represent heuristic decision-tree consensus proportions and **must not be interpreted as calibrated clinical probabilities**.

---

## 🎯 Educational Machine-Learning Workflow

The lab guides learners through five transparent, clearly labeled stages:

1. **Load Input Image**: Upload retinal fundus photography in common formats (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`) or select built-in synthetic educational samples.
2. **Preprocessing**:
   - **EXIF Orientation Correction**: Transposes rotated images according to EXIF metadata.
   - **Dark Border Cropping**: Automatically identifies and crops uninformative black circular camera margins.
   - **Standardized Resizing**: Scales the image to a standardized $256 \times 256$ pixels.
   - **CLAHE Enhancement**: Applies *Contrast Limited Adaptive Histogram Equalization* to enhance retinal blood vessels, exudates, and microaneurysms under uneven illumination.
3. **Feature Extraction**:
   - **HOG (Histogram of Oriented Gradients)**: Extracts directional edge and structural texture patterns (capturing vascular trees and lesion boundaries).
   - **RGB Color Histograms**: Computes 32-bin normalized color distributions per channel (capturing bright yellow lipid exudates, pale cotton-wool spots, and dark red hemorrhages).
   - Visualizes the 2D gradient orientation map and RGB channel histograms.
4. **Training and Evaluation**:
   - Accepts a user-uploaded labeled dataset ZIP or generates an in-memory 20-sample educational demo dataset.
   - Trains a `RandomForestClassifier` with balanced class weighting (`class_weight='balanced'`).
   - Uses a stratified train/test split with a fixed random seed (`random_state=42`) for reproducibility.
5. **Results & Prediction**:
   - Displays hold-out test accuracy, macro F1 score, training vs. test split counts, and dataset distribution.
   - Renders a confusion matrix heatmap and a per-class classification report (Precision, Sensitivity/Recall, F1-Score).
   - Predicts the severity grade for the loaded fundus image with confidence score breakdown.

---

## 📁 Dataset ZIP Structure

To train on your own data, upload a `.zip` file organized with numerical folder names corresponding to the 5 International Clinical Diabetic Retinopathy severity grades:

```text
dataset.zip
├── 0/   (No DR - Normal healthy fundus)
├── 1/   (Mild - Microaneurysms only)
├── 2/   (Moderate - Hard exudates, cotton-wool spots, microaneurysms)
├── 3/   (Severe - >20 intraretinal hemorrhages per quadrant, venous beading)
└── 4/   (Proliferative DR - Neovascularization, vitreous/preretinal hemorrhage)
```

### Supported Image Formats
- JPEG / JPG (`.jpg`, `.jpeg`)
- PNG (`.png`)
- BMP (`.bmp`)
- TIFF (`.tif`, `.tiff`)

*Note: For stratified evaluation, each class folder must contain at least 2 images so samples can be allocated to both training and test splits.*

---

## 💡 Image-Level vs. Patient-Level Split Consideration

In academic and clinical machine learning, data partition strategy is critical:
- **Image-Level Random Split**: When multiple photographs from the same patient (e.g., both eyes or multiple visits) are randomly scattered across train and test sets, **data leakage** occurs. The model can memorize individual patient-specific retinal pigmentation or anatomy, artificially inflating test accuracy.
- **Patient-Level Separation**: In clinical research, all images belonging to a single patient must strictly remain within either the training set or the test set (never split between both).

---

## 🚀 Setup and Run Instructions (Windows PowerShell)

This is a Python web application that runs directly in your browser through Streamlit. **It does not need to be compiled into an executable.**

Open **Windows PowerShell** in the project directory (`Retinal Disease Screening`) and execute the following commands:

```powershell
# 1. Create a Python virtual environment
py -m venv .venv

# 2. Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Install required packages
pip install -r requirements.txt

# 4. Launch the Streamlit web application
streamlit run app.py
```

*Note: If PowerShell execution policy restricts script activation, enable it for your current session using `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`.*

Once launched, your default web browser will automatically open to `http://localhost:8501`.

---

## 📦 Project Structure

```text
Retinal Disease Screening/
├── app.py                  # Main Streamlit single-page application & UI
├── image_processing.py     # EXIF correction, border crop, resize (256x256), CLAHE
├── feature_extraction.py   # HOG gradients and normalized RGB color histograms
├── dataset_utils.py        # ZIP validation, batch dataset processing & synthetic generator
├── model_utils.py          # Balanced Random Forest, stratified split & metrics
├── requirements.txt        # Python dependency manifest
└── README.md               # Documentation and execution guide
```

---

## 🛠️ Technology Stack

- **Frontend & App Framework**: [Streamlit](https://streamlit.io/)
- **Machine Learning**: [scikit-learn](https://scikit-learn.org/) (`RandomForestClassifier`, metrics)
- **Computer Vision & Image Processing**: [Pillow](https://python-pillow.org/), [scikit-image](https://scikit-image.org/), [NumPy](https://numpy.org/)
- **Visualizations**: [Matplotlib](https://matplotlib.org/)
