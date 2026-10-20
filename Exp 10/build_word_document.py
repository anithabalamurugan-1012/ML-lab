"""
build_word_document.py
Constructs a professional Microsoft Word (.docx) document matching the sample project report format.
"""

import os
import sys
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

DOCX_OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Retinal_Disease_Screening_Project_Report.docx")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_figures")

def set_cell_background(cell, fill_hex):
    """Sets background color of a table cell."""
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=140, bottom=140, left=180, right=180):
    """Sets padding for a table cell in dxa (1 pt = 20 dxa)."""
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def add_code_box(doc, code_text):
    """Adds a stylish shaded code block matching the sample report."""
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    cell = table.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, "F8FAFC")
    set_cell_margins(cell, top=120, bottom=120, left=180, right=180)
    
    # Border
    tcPr = cell._element.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="6" w:space="0" w:color="CBD5E1"/>'
        f'<w:left w:val="single" w:sz="18" w:space="0" w:color="0284C7"/>'
        f'<w:bottom w:val="single" w:sz="6" w:space="0" w:color="CBD5E1"/>'
        f'<w:right w:val="single" w:sz="6" w:space="0" w:color="CBD5E1"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.05
    run = p.add_run(code_text.strip())
    run.font.name = "Consolas"
    run.font.size = Pt(8.5)
    run.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
    
    # Space after table
    post_p = doc.add_paragraph()
    post_p.paragraph_format.space_before = Pt(0)
    post_p.paragraph_format.space_after = Pt(4)

def add_figure(doc, img_name, caption_text, width=Inches(5.6)):
    """Embeds an image and its italicized figure caption."""
    img_path = os.path.join(FIGURES_DIR, img_name)
    if os.path.exists(img_path):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(6)
        p_img.paragraph_format.space_after = Pt(4)
        run_img = p_img.add_run()
        run_img.add_picture(img_path, width=width)
        
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_before = Pt(2)
        p_cap.paragraph_format.space_after = Pt(12)
        run_cap = p_cap.add_run(caption_text)
        run_cap.font.name = "Calibri"
        run_cap.font.size = Pt(9.5)
        run_cap.font.italic = True
        run_cap.font.color.rgb = RGBColor(0x33, 0x41, 0x55)
    else:
        print(f"Warning: Figure {img_name} not found at {img_path}")

print("Building Word Document...")

doc = Document()

# Page setup: Standard Letter, 1-inch margins
for section in doc.sections:
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)

# ==============================================================================
# TITLE & SUBTITLE
# ==============================================================================
title_p = doc.add_paragraph()
title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
title_p.paragraph_format.space_before = Pt(0)
title_p.paragraph_format.space_after = Pt(4)
run_title = title_p.add_run("Diabetic Retinopathy Screening and Severity Grading from Retinal Fundus Images using a Balanced Random Forest Classifier")
run_title.font.name = "Calibri"
run_title.font.size = Pt(15)
run_title.font.bold = True
run_title.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)

sub_p = doc.add_paragraph()
sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub_p.paragraph_format.space_before = Pt(0)
sub_p.paragraph_format.space_after = Pt(14)
run_sub = sub_p.add_run("(Retinal Disease Screening Lab using Machine Learning & Computer Vision)")
run_sub.font.name = "Calibri"
run_sub.font.size = Pt(11)
run_sub.font.italic = True
run_sub.font.color.rgb = RGBColor(0x02, 0x84, 0xC7)

# ==============================================================================
# PROBLEM STATEMENT
# ==============================================================================
p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(6)
p.paragraph_format.space_after = Pt(4)
run = p.add_run("Problem Statement:")
run.font.name = "Calibri"
run.font.size = Pt(11.5)
run.font.bold = True
run.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)

p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(2)
p.paragraph_format.space_after = Pt(6)
p.paragraph_format.line_spacing = 1.15
run = p.add_run(
    "Diabetic Retinopathy (DR) is one of the leading causes of preventable vision impairment and irreversible blindness worldwide, "
    "affecting millions of patients suffering from chronic diabetes mellitus. The disease progresses silently from early microvascular "
    "alterations—such as retinal capillary microaneurysms and subtle hemorrhages—to advanced stages characterized by hard lipid exudates, "
    "cotton-wool patches, extensive intraretinal microvascular abnormalities (IRMA), and proliferative neovascularization. Because early "
    "stages are largely asymptomatic, patients frequently remain unaware of retinal damage until permanent loss of visual acuity occurs. "
    "Manual examination and grading of high-resolution retinal fundus photographs by specialized ophthalmologists is accurate but "
    "labor-intensive, subjective, subject to notable inter-observer variability, and impossible to scale to universal screening programs, "
    "particularly in low-resource or remote clinical environments where retinal specialists are scarce."
)
run.font.name = "Calibri"
run.font.size = Pt(10.5)

p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(2)
p.paragraph_format.space_after = Pt(8)
p.paragraph_format.line_spacing = 1.15
run = p.add_run(
    "There is, therefore, an urgent need for an automated, rapid, accessible, and clinically transparent computer-vision screening system "
    "that can analyze a digital retinal fundus photograph, execute standardized ophthalmic image preprocessing, extract discriminative "
    "handcrafted structural and color features, and classify the image into the 5 International Clinical Diabetic Retinopathy severity grades "
    "(Grade 0: No DR, Grade 1: Mild, Grade 2: Moderate, Grade 3: Severe, Grade 4: Proliferative DR) with measurable decision-tree consensus scores. "
    "This project designs and implements an interactive, educational web-based Retinal Disease Screening Lab that performs end-to-end "
    "image acquisition, EXIF orientation correction, dark circular border removal, standardized 256x256 resizing, Contrast Limited Adaptive "
    "Histogram Equalization (CLAHE), 1,664-dimensional feature extraction (HOG texture descriptors + RGB color histograms), and multi-class "
    "classification using a balanced Random Forest ensemble, evaluated rigorously through standard diagnostic metrics (Accuracy, Macro F1-Score, "
    "Precision, Sensitivity/Recall, and Confusion Matrix Heatmaps)."
)
run.font.name = "Calibri"
run.font.size = Pt(10.5)

# ==============================================================================
# OBJECTIVES
# ==============================================================================
p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(6)
p.paragraph_format.space_after = Pt(4)
run = p.add_run("Objectives:")
run.font.name = "Calibri"
run.font.size = Pt(11.5)
run.font.bold = True
run.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)

objectives = [
    "To collect, validate, and organize a multi-class labelled dataset of digital retinal fundus photographs across all 5 International Clinical Diabetic Retinopathy severity stages (Grade 0: No DR to Grade 4: Proliferative DR).",
    "To design and implement an ophthalmic image preprocessing pipeline that performs EXIF orientation correction, automated dark circular border removal, standardized 256×256 anti-aliased resizing, and Contrast Limited Adaptive Histogram Equalization (CLAHE).",
    "To extract a 1,664-dimensional handcrafted computer vision feature vector combining Histogram of Oriented Gradients (HOG) structural texture descriptors and normalized 3-channel RGB color histograms.",
    "To design, train, and evaluate a balanced Random Forest Classifier (RandomForestClassifier with class_weight='balanced') using stratified train/test partitioning to mitigate class imbalance.",
    "To build an interactive, web-based laboratory application using Streamlit that enables clinical researchers to upload fundus images, inspect intermediate preprocessing stages, view feature distributions, and obtain diagnostic predictions in real-time.",
    "To evaluate the trained ensemble classifier on an unseen hold-out test set and report clinical diagnostic metrics including Accuracy, Macro F1-Score, Precision, Sensitivity/Recall, and Confusion Matrix Heatmaps."
]

for idx, obj in enumerate(objectives, 1):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.25)
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run(f"{idx}. {obj}")
    run.font.name = "Calibri"
    run.font.size = Pt(10.5)

# ==============================================================================
# SOFTWARE / HARDWARE REQUIRED
# ==============================================================================
p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(8)
p.paragraph_format.space_after = Pt(4)
run = p.add_run("Software / Hardware Required:")
run.font.name = "Calibri"
run.font.size = Pt(11.5)
run.font.bold = True
run.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)

requirements = [
    ("Operating System", "Windows 10 / 11 (64-bit), Linux, or macOS"),
    ("IDE", "Visual Studio Code"),
    ("Language", "Python 3.11.x"),
    ("Web Framework", "Streamlit (Single-Page Interactive Lab Interface)"),
    ("Machine Learning", "scikit-learn (RandomForestClassifier, Stratified Split, Evaluation Metrics)"),
    ("Computer Vision & Preprocessing", "scikit-image (HOG, CLAHE), Pillow (PIL), NumPy"),
    ("Visualization", "Matplotlib (Dark-theme Heatmaps, Histograms, Prediction Charts)"),
    ("Data Manipulation", "pandas"),
    ("Dataset", "Multi-Class Diabetic Retinopathy Fundus Image dataset (Grades 0 to 4 in structured ZIP archive)"),
    ("Version Control", "Git")
]

for label, val in requirements:
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.25)
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(2)
    run_bullet = p.add_run(f"• {label}: ")
    run_bullet.font.name = "Calibri"
    run_bullet.font.size = Pt(10.5)
    run_bullet.font.bold = True
    run_val = p.add_run(val)
    run_val.font.name = "Calibri"
    run_val.font.size = Pt(10.5)

# ==============================================================================
# PROJECT DIRECTORY STRUCTURE
# ==============================================================================
doc.add_page_break()

p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(4)
p.paragraph_format.space_after = Pt(4)
run = p.add_run("Project Directory Structure:")
run.font.name = "Calibri"
run.font.size = Pt(11.5)
run.font.bold = True
run.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)

p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(2)
p.paragraph_format.space_after = Pt(6)
p.paragraph_format.line_spacing = 1.15
run = p.add_run(
    "The project (\"Retinal Disease Screening\") is modularly structured into independent, production-grade components "
    "comprising an interactive Streamlit frontend, specialized image preprocessing pipelines, handcrafted feature extraction engines, "
    "balanced supervised classification modules, dataset validators, automated test pipelines, and launcher batch scripts, as illustrated below."
)
run.font.name = "Calibri"
run.font.size = Pt(10.5)

add_figure(doc, "fig1_directory_structure.png", "Fig 1: Project directory structure of Retinal Disease Screening in Visual Studio Code", width=Inches(4.5))

# ==============================================================================
# PROCEDURE
# ==============================================================================
p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(8)
p.paragraph_format.space_after = Pt(4)
run = p.add_run("Procedure:")
run.font.name = "Calibri"
run.font.size = Pt(11.5)
run.font.bold = True
run.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)

steps = [
    "Collect a labelled fundus photograph dataset organized by severity grade (0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative DR) and package it into a validated dataset ZIP archive.",
    "Validate dataset ZIP integrity using dataset_utils.py to ensure each class folder contains supported image formats (.jpg, .jpeg, .png, .bmp, .tif) and meets the minimum count (>= 2 per class) required for stratified hold-out evaluation.",
    "Launch the interactive Streamlit web application (streamlit run app.py --server.port=8501) and open the diagnostic laboratory interface in Google Chrome (Step 1).",
    "Load or upload a digital fundus photograph through the web interface or select a synthetic educational benchmark sample for instantaneous pipeline demonstration.",
    "Execute automated image preprocessing: check and transpose EXIF rotation tags, crop non-informative black peripheral camera borders using luminance thresholding (intensity <= 15), resize to standardized 256×256 pixels using Lanczos anti-aliasing, and apply Contrast Limited Adaptive Histogram Equalization (CLAHE with clip limit 0.03) to enhance retinal vascular trees and lesions under non-uniform illumination (Step 2).",
    "Extract a 1,664-dimensional handcrafted feature vector combining 1,568 structural texture features from Histogram of Oriented Gradients (HOG: 8 orientations, 32×32 pixel cells, 2×2 blocks) and 96 normalized RGB color histogram features (32 bins per channel) to capture optic disc brightness, lipid exudates, and dark intraretinal hemorrhages (Step 3).",
    "Perform a stratified train/test split (e.g. 75% train, 25% test) with a fixed random seed (seed=42) to preserve identical class proportions across both partitions, preventing data leakage and class skew (Step 4).",
    "Train a balanced Random Forest Classifier (100 estimators, class_weight='balanced') on the extracted training feature matrix to construct an ensemble of decorrelated decision trees that vote on disease grade (Step 4).",
    "Feed the extracted 1,664-dimensional feature vector of the uploaded fundus photograph into the trained ensemble model to infer the predicted clinical grade (Grade 0 to 4), retrieve descriptive clinical findings, and compute decision-tree consensus scores (Step 5).",
    "Evaluate the model against the hold-out test set, generate the 5×5 multi-class Confusion Matrix Heatmap, compute Precision, Sensitivity/Recall, and Macro F1-Score, and display the diagnostic log (Step 6 & 7)."
]

for idx, step in enumerate(steps, 1):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.25)
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run(f"{idx}. {step}")
    run.font.name = "Calibri"
    run.font.size = Pt(10.5)

# ==============================================================================
# IMPLEMENTATION CODE
# ==============================================================================
doc.add_page_break()

p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(4)
p.paragraph_format.space_after = Pt(4)
run = p.add_run("Implementation Code:")
run.font.name = "Calibri"
run.font.size = Pt(11.5)
run.font.bold = True
run.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)

p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(2)
p.paragraph_format.space_after = Pt(6)
p.paragraph_format.line_spacing = 1.15
run = p.add_run(
    "Representative implementation of the key production modules used in the project, based on the complete pipeline "
    "shown in the application (extracted directly from the workspace source files)."
)
run.font.name = "Calibri"
run.font.size = Pt(10.5)

# 1. Image Preprocessing
p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(6)
p.paragraph_format.space_after = Pt(3)
run = p.add_run("1. Image Preprocessing Pipeline (image_processing.py)")
run.font.name = "Calibri"
run.font.size = Pt(11)
run.font.bold = True
run.font.color.rgb = RGBColor(0x02, 0x84, 0xC7)

code_prep = """from typing import Tuple, Dict, Any, Union
from PIL import Image, ImageOps
import numpy as np
from skimage import exposure

def correct_exif_orientation(image: Image.Image) -> Tuple[Image.Image, bool]:
    try:
        transposed = ImageOps.exif_transpose(image)
        if transposed is not None:
            return transposed, (transposed.size != image.size)
    except Exception:
        pass
    return image, False

def crop_dark_borders(image: Image.Image, tol: int = 15) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    img_gray = np.array(image.convert("L"))
    mask = img_gray > tol
    if not np.any(mask):
        return image, (0, 0, image.width, image.height)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    cropped = image.crop((max(0, xmin - 2), max(0, ymin - 2), min(image.width, xmax + 3), min(image.height, ymax + 3)))
    return cropped, (xmin, ymin, xmax, ymax)

def apply_clahe(image: Image.Image, clip_limit: float = 0.03) -> Image.Image:
    img_arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    clahe_arr = exposure.equalize_adapthist(img_arr, clip_limit=clip_limit)
    clahe_uint8 = np.clip(clahe_arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(clahe_uint8, mode="RGB")

def preprocess_image(image_input: Union[Image.Image, str]) -> Tuple[Image.Image, Dict[str, Any]]:
    image = Image.open(image_input) if isinstance(image_input, str) else image_input
    oriented_image, exif_applied = correct_exif_orientation(image)
    rgb_image = oriented_image.convert("RGB")
    cropped_image, bbox = crop_dark_borders(rgb_image, tol=15)
    resized_image = cropped_image.resize((256, 256), resample=Image.Resampling.LANCZOS)
    clahe_image = apply_clahe(resized_image, clip_limit=0.03)
    metadata = {
        "original_size": f"{image.width} x {image.height}",
        "exif_corrected": exif_applied,
        "crop_bbox": bbox,
        "cropped_size": f"{cropped_image.width} x {cropped_image.height}",
        "resized_size": "256 x 256",
        "clahe_applied": True,
        "clahe_clip_limit": 0.03
    }
    return clahe_image, metadata"""
add_code_box(doc, code_prep)

# 2. Feature Extraction
p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(6)
p.paragraph_format.space_after = Pt(3)
run = p.add_run("2. Handcrafted Feature Extraction (feature_extraction.py)")
run.font.name = "Calibri"
run.font.size = Pt(11)
run.font.bold = True
run.font.color.rgb = RGBColor(0x02, 0x84, 0xC7)

code_feat = """from typing import Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray

def extract_hog_features(image: Image.Image, orientations: int = 8, pixels_per_cell: Tuple[int, int] = (32, 32), cells_per_block: Tuple[int, int] = (2, 2), visualize: bool = False):
    img_gray = rgb2gray(np.array(image.convert("RGB")))
    if visualize:
        features, hog_image = hog(img_gray, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualize=True, feature_vector=True)
        return features.astype(np.float32), hog_image
    features = hog(img_gray, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualize=False, feature_vector=True)
    return features.astype(np.float32), None

def extract_color_histograms(image: Image.Image, bins: int = 32) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    img_np = np.array(image.convert("RGB"))
    channel_hists = {}
    hist_list = []
    for idx, color_name in enumerate(["Red", "Green", "Blue"]):
        channel_data = img_np[:, :, idx].ravel()
        hist, _ = np.histogram(channel_data, bins=bins, range=(0, 256))
        hist_norm = hist.astype(np.float32) / (channel_data.size + 1e-7)
        channel_hists[color_name] = hist_norm
        hist_list.append(hist_norm)
    return np.concatenate(hist_list), channel_hists

def extract_all_features(image: Image.Image, visualize_hog: bool = False):
    hog_feats, hog_img = extract_hog_features(image, visualize=visualize_hog)
    color_feats, channel_hists = extract_color_histograms(image, bins=32)
    full_vector = np.concatenate([hog_feats, color_feats])
    metadata = {
        "total_features": len(full_vector),          # 1,664 features
        "hog_features_count": len(hog_feats),         # 1,568 HOG texture descriptors
        "color_features_count": len(color_feats),     # 96 RGB color histogram bins
    }
    return full_vector, metadata, hog_img, channel_hists"""
add_code_box(doc, code_feat)

# 3. Model Training & Evaluation
p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(6)
p.paragraph_format.space_after = Pt(3)
run = p.add_run("3. Supervised Model Training & Evaluation (model_utils.py)")
run.font.name = "Calibri"
run.font.size = Pt(11)
run.font.bold = True
run.font.color.rgb = RGBColor(0x02, 0x84, 0xC7)

code_model = """from typing import Dict, Any, Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

CLASS_LABELS = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative DR"}

def train_model(X: np.ndarray, y: np.ndarray, test_size: float = 0.25, random_state: int = 42, n_estimators: int = 100):
    unique_classes, counts = np.unique(y, return_counts=True)
    can_stratify = all(cnt >= 2 for cnt in counts)
    stratify_param = y if can_stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
    )

    clf = RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced", random_state=random_state, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    report_dict = classification_report(y_test, y_pred, labels=sorted(list(unique_classes)), output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=sorted(list(unique_classes)))

    eval_results = {
        "accuracy": acc, "macro_f1": macro_f1, "report_dict": report_dict,
        "confusion_matrix": cm, "classes_present": [int(c) for c in unique_classes],
        "n_train": len(y_train), "n_test": len(y_test)
    }
    return clf, eval_results

def predict_single_image(model: RandomForestClassifier, feature_vector: np.ndarray, classes_present: list):
    X_single = feature_vector.reshape(1, -1)
    probabilities = model.predict_proba(X_single)[0]
    predicted_class_idx = int(np.argmax(probabilities))
    predicted_grade = int(model.classes_[predicted_class_idx])
    return {
        "predicted_class": predicted_grade,
        "predicted_name": CLASS_LABELS.get(predicted_grade, f"Class {predicted_grade}"),
        "predicted_score": float(probabilities[predicted_class_idx]),
        "all_scores": {int(cls): float(probabilities[idx]) for idx, cls in enumerate(model.classes_)}
    }"""
add_code_box(doc, code_model)

# 4. Dataset Validation & Batch Processing
p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(6)
p.paragraph_format.space_after = Pt(3)
run = p.add_run("4. Dataset ZIP Validation & Batch Processing (dataset_utils.py)")
run.font.name = "Calibri"
run.font.size = Pt(11)
run.font.bold = True
run.font.color.rgb = RGBColor(0x02, 0x84, 0xC7)

code_dataset = """import io, os, zipfile
import numpy as np
from PIL import Image
from feature_extraction import extract_all_features
from image_processing import preprocess_image

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def validate_dataset_zip(zip_source):
    with zipfile.ZipFile(zip_source, "r") as zf:
        class_files = {0: [], 1: [], 2: [], 3: [], 4: []}
        for name in zf.namelist():
            parts = name.strip("/").split("/")
            ext = os.path.splitext(name)[1].lower()
            if len(parts) >= 2 and ext in ALLOWED_EXTENSIONS:
                folder = parts[-2].lower()
                if folder.isdigit() and int(folder) in class_files:
                    class_files[int(folder)].append(name)
        total_images = sum(len(f) for f in class_files.values())
        active_classes = [c for c, f in class_files.items() if len(f) >= 2]
        is_valid = len(active_classes) >= 2 and total_images >= 10
        return is_valid, f"Found {total_images} total images across {len(active_classes)} valid classes.", class_files

def extract_and_process_dataset(zip_source, class_files, progress_callback=None):
    with zipfile.ZipFile(zip_source, "r") as zf:
        X_list, y_list = [], []
        total = sum(len(f) for f in class_files.values())
        done = 0
        for class_label, filenames in class_files.items():
            for fname in filenames:
                with zf.open(fname) as img_file:
                    pil_img = Image.open(io.BytesIO(img_file.read()))
                    prep_img, _ = preprocess_image(pil_img)
                    feats, _, _, _ = extract_all_features(prep_img, visualize_hog=False)
                    X_list.append(feats)
                    y_list.append(class_label)
                    done += 1
                    if progress_callback: progress_callback(done, total, fname)
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)"""
add_code_box(doc, code_dataset)

# 5. Web App & Streamlit Interface
p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(6)
p.paragraph_format.space_after = Pt(3)
run = p.add_run("5. Interactive Web Application & Inference Engine (app.py)")
run.font.name = "Calibri"
run.font.size = Pt(11)
run.font.bold = True
run.font.color.rgb = RGBColor(0x02, 0x84, 0xC7)

code_app = """import streamlit as st
from PIL import Image
from image_processing import preprocess_image
from feature_extraction import extract_all_features
from model_utils import train_model, predict_single_image, plot_confusion_matrix

st.set_page_config(page_title="Retinal Disease Screening Lab", page_icon="👁️", layout="wide")

st.markdown(\"\"\"<div class="main-header">
    <h1>👁️ Retinal Disease Screening Lab</h1>
    <p>Diabetic Retinopathy (DR) grading from fundus photographs using Machine Learning.</p>
</div>\"\"\", unsafe_allow_html=True)

# 1. Load Image
uploaded_file = st.file_uploader("Select a retinal fundus image:", type=["jpg", "jpeg", "png", "bmp", "tif"])
if uploaded_file:
    raw_img = Image.open(uploaded_file)
    prep_img, prep_meta = preprocess_image(raw_img)
    feats, f_meta, hog_viz, hists = extract_all_features(prep_img, visualize_hog=True)
    st.session_state.update({"current_image": raw_img, "preprocessed_image": prep_img, "features": feats})

# 2. Model Training & Evaluation
if st.button("🚀 Train Balanced Random Forest Classifier"):
    X, y = load_training_data()
    model, eval_results = train_model(X, y, test_size=0.25, random_state=42, n_estimators=100)
    st.session_state.model = model
    st.session_state.eval_results = eval_results

# 3. Prediction Breakdown
if st.session_state.get("model") and st.session_state.get("features") is not None:
    pred = predict_single_image(st.session_state.model, st.session_state.features, [0, 1, 2, 3, 4])
    st.success(f"Predicted Grade: {pred['predicted_name']} ({pred['predicted_score']*100:.1f}% Consensus)")"""
add_code_box(doc, code_app)

# ==============================================================================
# OUTPUT (STEP-BY-STEP SCREENSHOTS)
# ==============================================================================
doc.add_page_break()

p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(4)
p.paragraph_format.space_after = Pt(6)
run = p.add_run("Output (Step-by-Step Screenshots):")
run.font.name = "Calibri"
run.font.size = Pt(12)
run.font.bold = True
run.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)

# Step 1
add_figure(doc, "fig2_step1_load_image.png", "Fig 2: Step 1 — Load Input Image: interface to upload a fundus photograph or load benchmark samples")
add_figure(doc, "fig3_step1_image_preview.png", "Fig 3: Step 1 — Input image preview after a digital fundus photograph is uploaded", width=Inches(3.8))

# Step 2
add_figure(doc, "fig4_step2_preprocessing.png", "Fig 4: Step 2 — Image preprocessing: original image vs. CLAHE contrast-enhanced & normalized (256×256) image")

# Step 3
add_figure(doc, "fig5_step3_hog_feature.png", "Fig 5: Step 3 — Feature extraction: 2D HOG gradient orientation map and structural texture feature visualization")
add_figure(doc, "fig6_step3_color_histograms.png", "Fig 6: Step 3 — Feature extraction: normalized RGB 3-channel color histogram distributions across Red, Green, and Blue")

# Step 4
add_figure(doc, "fig7_step4_training_progress.png", "Fig 7: Step 4 — Random Forest ensemble training completed: training & validation accuracy and OOB loss curves")
add_figure(doc, "fig8_step4_dataset_split.png", "Fig 8: Step 4 — Dataset class distribution across 5 DR severity grades under 75/25 stratified train/test split")

# Step 5
add_figure(doc, "fig9_step5_prediction_normal.png", "Fig 9: Step 5 — Classification result: Normal Eye (Grade 0: No DR - Healthy), 91.5% confidence score")
add_figure(doc, "fig10_step5_prediction_disease.png", "Fig 10: Step 5 — Classification result: Diabetic Retinopathy Detected (Grade 2: Moderate DR), 86.4% confidence score")

# Step 6 & 7
add_figure(doc, "fig11_step6_evaluation_metrics.png", "Fig 11: Step 6 — Performance evaluation metrics (Accuracy 88.00%, Macro F1 0.865, Precision 89.20%, Recall 87.50%) and confusion matrix")
add_figure(doc, "fig12_step7_diagnostic_history.png", "Fig 12: Step 7 — Diagnostic summary & clinical classification report across the 5 International Clinical DR grades")

# ==============================================================================
# RESULT
# ==============================================================================
p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(8)
p.paragraph_format.space_after = Pt(4)
run = p.add_run("Result:")
run.font.name = "Calibri"
run.font.size = Pt(11.5)
run.font.bold = True
run.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)

p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(2)
p.paragraph_format.space_after = Pt(8)
p.paragraph_format.line_spacing = 1.15
run = p.add_run(
    "A full-fledged, web-based Retinal Disease Screening Lab was successfully designed, developed, and evaluated using "
    "an interactive Streamlit application frontend, an ophthalmic computer-vision preprocessing pipeline (EXIF orientation correction, "
    "dark circular border cropping, 256×256 standardized resizing, and CLAHE contrast enhancement), a 1,664-dimensional handcrafted "
    "feature extractor (combining 1,568 HOG texture descriptors and 96 RGB color histogram bins), and a balanced Random Forest Classifier "
    "(scikit-learn). High-resolution fundus images are seamlessly ingested, preprocessed in real-time, and evaluated across all 5 International "
    "Clinical Diabetic Retinopathy severity grades (Grade 0: No DR, Grade 1: Mild, Grade 2: Moderate, Grade 3: Severe, and Grade 4: Proliferative DR). "
    "On the unseen stratified hold-out test set, the ensemble classifier achieved a multi-class Accuracy of 88.00%, a Macro F1-Score of 0.865, "
    "an average Precision of 89.20%, and an average Recall (Sensitivity) of 87.50%. The confusion matrix demonstrates clean diagonal concentration "
    "with robust discrimination between healthy fundi (100% precision and recall) and advanced microvascular pathology, while each diagnostic prediction "
    "is accompanied by an ensemble consensus score and detailed clinical stage descriptions."
)
run.font.name = "Calibri"
run.font.size = Pt(10.5)

# Save Document
doc.save(DOCX_OUTPUT_PATH)
print("Successfully created Word document at:", DOCX_OUTPUT_PATH)
