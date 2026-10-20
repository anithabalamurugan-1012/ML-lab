"""
app.py
======
Retinal Disease Screening Lab
An educational diabetic retinopathy image-classification web application.

Workflow:
1. Load a retinal fundus image
2. Preprocess the image (EXIF orientation, dark border crop, 256x256 resize, CLAHE)
3. Extract features (HOG texture + RGB color histograms)
4. Train & evaluate a balanced RandomForestClassifier on a labeled dataset ZIP
5. Display performance metrics (accuracy, macro F1, report, confusion matrix) & inference results
"""

import io
import os
import sys

# Ensure local modules can be imported regardless of execution context
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st

from image_processing import preprocess_image
from feature_extraction import extract_all_features
from dataset_utils import (
    CLASS_LABELS,
    validate_dataset_zip,
    extract_and_process_dataset,
    create_sample_dataset_zip_bytes,
    generate_synthetic_fundus_image
)
from model_utils import (
    CLASS_DESCRIPTIONS,
    train_model,
    plot_confusion_matrix,
    predict_single_image
)

# Page configuration
st.set_page_config(
    page_title="Retinal Disease Screening Lab",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern educational interface
st.markdown("""
<style>
    /* Global styles */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0369a1 100%);
        padding: 1.8rem 2rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
        color: #f8fafc;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        color: #f0f9ff;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        color: #94a3b8;
        font-size: 1.05rem;
    }
    .section-badge {
        display: inline-flex;
        align-items: center;
        background: #0284c7;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .step-box {
        background: #1e293b;
        border-radius: 12px;
        padding: 1.25rem;
        border: 1px solid #334155;
        margin-bottom: 1rem;
    }
    .disclaimer-banner {
        background: #451a03;
        border-left: 5px solid #f59e0b;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        color: #fef3c7;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .info-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 10px;
        padding: 1rem;
        margin-top: 0.5rem;
    }
    .metric-card {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .grade-badge-0 { background-color: #059669; color: white; padding: 4px 10px; border-radius: 6px; font-weight: 600; }
    .grade-badge-1 { background-color: #0284c7; color: white; padding: 4px 10px; border-radius: 6px; font-weight: 600; }
    .grade-badge-2 { background-color: #d97706; color: white; padding: 4px 10px; border-radius: 6px; font-weight: 600; }
    .grade-badge-3 { background-color: #ea580c; color: white; padding: 4px 10px; border-radius: 6px; font-weight: 600; }
    .grade-badge-4 { background-color: #dc2626; color: white; padding: 4px 10px; border-radius: 6px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "model" not in st.session_state:
    st.session_state.model = None
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "preprocessed_image" not in st.session_state:
    st.session_state.preprocessed_image = None
if "prep_meta" not in st.session_state:
    st.session_state.prep_meta = None
if "features" not in st.session_state:
    st.session_state.features = None
if "feature_meta" not in st.session_state:
    st.session_state.feature_meta = None
if "hog_viz" not in st.session_state:
    st.session_state.hog_viz = None
if "color_hists" not in st.session_state:
    st.session_state.color_hists = None

# Header
st.markdown("""
<div class="main-header">
    <h1>👁️ Retinal Disease Screening Lab</h1>
    <p>An interactive, educational machine-learning pipeline for Diabetic Retinopathy (DR) grading from fundus photographs.</p>
</div>
""", unsafe_allow_html=True)

# Educational Disclaimer Banner
st.markdown("""
<div class="disclaimer-banner">
    <strong>⚠️ Educational Prototype Disclaimer:</strong> This web application is designed strictly for classroom, academic, and demonstration purposes.
    It has not been clinically validated or approved by regulatory bodies (e.g. FDA, CE-IVD).
    <strong>It must NOT be used for medical diagnosis, clinical decision support, or treatment planning.</strong>
</div>
""", unsafe_allow_html=True)

# Sidebar with Educational Guidelines & Demo tools
with st.sidebar:
    st.header("📚 Lab Guide & Controls")
    st.markdown("""
    This lab implements the full computer-vision machine-learning workflow:
    1. **Fundus Image Acquisition**
    2. **Ophthalmic Preprocessing**
    3. **Handcrafted Feature Engineering**
    4. **Supervised Model Training**
    5. **Evaluation & Severity Grading**
    """)

    st.divider()
    st.subheader("🎓 Diabetic Retinopathy Stages")
    for grade, name in CLASS_LABELS.items():
        st.markdown(f"**Grade {grade}**: *{name}*")
        st.caption(CLASS_DESCRIPTIONS[grade])

    st.divider()
    st.subheader("🧪 Need a Sample Dataset?")
    st.markdown("Don't have a dataset ZIP ready? You can download or load our educational synthetic dataset.")
    sample_zip_bytes = create_sample_dataset_zip_bytes(images_per_class=4)
    st.download_button(
        label="📥 Download Sample Dataset ZIP",
        data=sample_zip_bytes,
        file_name="retinal_dr_sample_dataset.zip",
        mime="application/zip",
        help="Download a balanced educational dataset ZIP with 20 fundus images (4 per class 0-4)."
    )

    st.divider()
    st.markdown("### 💡 Split Separation Note")
    st.caption(
        "**Image-level vs. Patient-level Splits**: "
        "Standard image-level random splits can overestimate real-world diagnostic performance if multiple images from the same eye or patient appear in both train and test splits (data leakage). "
        "For serious medical evaluations, strict patient-level separation is required."
    )

# Main Application Layout: 5 Sections

# ==============================================================================
# SECTION 1: Load Input Image
# ==============================================================================
st.markdown("### 1. Load Input Image")
st.markdown("""
<div class="info-card">
    <strong>Why this step matters:</strong> Retinal fundus photography captures the interior surface of the eye (retina, optic disc, macula, and retinal blood vessels). In clinical practice, high-resolution fundus cameras take digital photographs in various formats (JPG, PNG, TIFF).
</div>
""", unsafe_allow_html=True)

col_upload, col_preview = st.columns([1, 1], gap="large")

with col_upload:
    uploaded_file = st.file_uploader(
        "Select a retinal fundus image:",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        help="Upload a retinal fundus photograph in JPG, JPEG, PNG, BMP, or TIFF format."
    )

    st.caption("Or load a preset synthetic fundus sample for demonstration:")
    sample_choice = st.selectbox(
        "Preset Fundus Sample:",
        options=[
            "None (use uploaded file)",
            "Sample 0: Healthy Retina (Grade 0 - No DR)",
            "Sample 1: Mild DR (Grade 1 - Microaneurysms)",
            "Sample 2: Moderate DR (Grade 2 - Exudates & MAs)",
            "Sample 3: Severe DR (Grade 3 - Hemorrhages & CWS)",
            "Sample 4: Proliferative DR (Grade 4 - Neovascularization)"
        ]
    )

    # Process selected input
    raw_pil_image = None
    if uploaded_file is not None:
        try:
            raw_pil_image = Image.open(uploaded_file)
            st.success(f"Loaded '{uploaded_file.name}' ({raw_pil_image.width} × {raw_pil_image.height} px, {raw_pil_image.format})")
        except Exception as e:
            st.error(f"Error loading uploaded image: {str(e)}")
            raw_pil_image = None
    elif sample_choice != "None (use uploaded file)":
        grade_selected = int(sample_choice.split(":")[0].replace("Sample ", ""))
        raw_pil_image = generate_synthetic_fundus_image(class_label=grade_selected, seed=42 + grade_selected)
        st.info(f"Loaded educational synthetic fundus sample: {CLASS_LABELS[grade_selected]}")

    if raw_pil_image is not None:
        st.session_state.current_image = raw_pil_image
        # Trigger preprocessing automatically on new image load
        prep_img, prep_meta = preprocess_image(raw_pil_image)
        st.session_state.preprocessed_image = prep_img
        st.session_state.prep_meta = prep_meta

        # Trigger feature extraction
        feats, f_meta, hog_viz, hists = extract_all_features(prep_img, visualize_hog=True)
        st.session_state.features = feats
        st.session_state.feature_meta = f_meta
        st.session_state.hog_viz = hog_viz
        st.session_state.color_hists = hists

with col_preview:
    if st.session_state.current_image is not None:
        st.image(
            st.session_state.current_image,
            caption=f"Original Input Image ({st.session_state.current_image.width} × {st.session_state.current_image.height} px)",
            use_container_width=True
        )
    else:
        st.info("👈 Please upload a fundus photograph or choose an educational sample above to preview.")

st.divider()

# ==============================================================================
# SECTION 2: Preprocessing
# ==============================================================================
st.markdown("### 2. Preprocessing")
st.markdown("""
<div class="info-card">
    <strong>Why this step matters:</strong> Raw fundus images frequently contain uninformative black circular camera borders, orientation discrepancies, non-uniform field illumination, and varying resolutions. Preprocessing cleans and standardizes the image so downstream feature extractors can reliably detect subtle retinal micro-lesions.
</div>
""", unsafe_allow_html=True)

if st.session_state.preprocessed_image is not None and st.session_state.prep_meta is not None:
    meta = st.session_state.prep_meta
    col_prep_view, col_prep_steps = st.columns([1, 1], gap="large")

    with col_prep_view:
        st.image(
            st.session_state.preprocessed_image,
            caption="Preprocessed Image (256 × 256 px, CLAHE Enhanced)",
            use_container_width=True
        )

    with col_prep_steps:
        st.markdown("#### Applied Preprocessing Steps:")
        
        st.markdown(f"""
        - **1. EXIF Orientation Transpose**:
          Ensures the eye image is upright based on EXIF tag metadata.
          *(Result: {'Corrected' if meta.get('exif_corrected') else 'Already correctly oriented'})*
        
        - **2. Dark Border Cropping**:
          Automatically detects camera aperture edges by thresholding dark background margins (intensity &le; 15).
          *(Original: `{meta.get('original_size')}` &rarr; Cropped: `{meta.get('cropped_size')}`)*
        
        - **3. Standardized Resizing**:
          Bicubic / Lanczos resampling to uniform **256 &times; 256** pixels to maintain equal feature vector lengths across varying camera resolutions.
        
        - **4. CLAHE Contrast Enhancement**:
          *Contrast Limited Adaptive Histogram Equalization* (clip limit: `{meta.get('clahe_clip_limit', 0.03)}`).
          Enhances local contrast across retinal vascular trees, bright lipid exudates, and dark intraretinal microaneurysms while preventing noise over-amplification in uniform areas.
        """)
        st.success("✅ Image successfully standardized and ready for feature extraction.")
else:
    st.info("Awaiting input image to execute preprocessing pipeline.")

st.divider()

# ==============================================================================
# SECTION 3: Feature Extraction
# ==============================================================================
st.markdown("### 3. Feature Extraction")
st.markdown("""
<div class="info-card">
    <strong>Why this step matters:</strong> Machine-learning algorithms like Random Forests require structured numerical vectors rather than raw pixel matrices. We combine <em>structural texture features</em> (HOG) with <em>color distribution features</em> (RGB histograms) to characterize vascular abnormalities, hemorrhages, and lipid exudates.
</div>
""", unsafe_allow_html=True)

if st.session_state.features is not None and st.session_state.feature_meta is not None:
    f_meta = st.session_state.feature_meta

    # Metrics row for feature counts
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{f_meta['total_features']:,}</div>
            <div class="metric-label">Total Feature Vector Length</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{f_meta['hog_features_count']:,}</div>
            <div class="metric-label">HOG Texture Features</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{f_meta['color_features_count']:,}</div>
            <div class="metric-label">RGB Color Histogram Features</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    col_hog_disp, col_hist_disp = st.columns([1, 1], gap="large")

    with col_hog_disp:
        st.markdown("#### Feature Type A: HOG (Histogram of Oriented Gradients)")
        st.caption(
            "HOG captures edge directions and structural contours. In fundus screening, HOG quantifies blood vessel tortuosity, optic disc boundary sharpness, and textural disruptions caused by lesions."
        )
        if st.session_state.hog_viz is not None:
            # Render HOG visualization
            fig_hog, ax_hog = plt.subplots(figsize=(4, 4), dpi=100)
            fig_hog.patch.set_facecolor("#0F172A")
            ax_hog.set_facecolor("#0F172A")
            ax_hog.imshow(st.session_state.hog_viz, cmap="magma")
            ax_hog.axis("off")
            plt.tight_layout()
            st.pyplot(fig_hog, use_container_width=True)
            plt.close(fig_hog)
            st.caption(f"Config: {f_meta['hog_orientations']} orientations, cell size {f_meta['hog_cell_size']}, block size {f_meta['hog_block_size']}")

    with col_hist_disp:
        st.markdown("#### Feature Type B: RGB Color Histograms")
        st.caption(
            "Captures chromatic signatures across the retina: bright yellow lipid deposits (exudates), pale cotton-wool spots, warm retinal background pigment, and dark red hemorrhages."
        )
        if st.session_state.color_hists is not None:
            fig_hist, ax_hist = plt.subplots(figsize=(6, 4.2), dpi=100)
            fig_hist.patch.set_facecolor("#0F172A")
            ax_hist.set_facecolor("#1E293B")
            
            colors_map = {"Red": "#EF4444", "Green": "#10B981", "Blue": "#3B82F6"}
            for channel, hist_vals in st.session_state.color_hists.items():
                ax_hist.plot(hist_vals, label=f"{channel} Channel", color=colors_map[channel], lw=2)

            ax_hist.set_title("Normalized Color Histograms (32 bins/channel)", color="#F8FAFC", fontsize=11)
            ax_hist.set_xlabel("Intensity Bin", color="#94A3B8", fontsize=9)
            ax_hist.set_ylabel("Normalized Frequency", color="#94A3B8", fontsize=9)
            ax_hist.tick_params(colors="#CBD5E1")
            ax_hist.grid(True, linestyle="--", alpha=0.2, color="#94A3B8")
            ax_hist.legend(facecolor="#0F172A", edgecolor="#334155", labelcolor="#F8FAFC", fontsize=8)
            for spine in ax_hist.spines.values():
                spine.set_edgecolor("#334155")
            plt.tight_layout()
            st.pyplot(fig_hist, use_container_width=True)
            plt.close(fig_hist)
            st.caption("32 bins per channel &times; 3 channels = 96 color distribution values.")
else:
    st.info("Awaiting preprocessed image to extract feature vectors.")

st.divider()

# ==============================================================================
# SECTION 4: Training and Evaluation
# ==============================================================================
st.markdown("### 4. Training and Evaluation")
st.markdown("""
<div class="info-card">
    <strong>Why this step matters:</strong> In supervised machine learning, we train an ensemble model (Random Forest) on labeled fundus images representing the 5 diabetic retinopathy grades.
    A <em>stratified train/test split</em> reserves unseen hold-out samples while preserving class balance.
</div>
""", unsafe_allow_html=True)

st.markdown("""
**Dataset Archive Requirements:**
- Upload a `.zip` archive containing class folders named **`0`**, **`1`**, **`2`**, **`3`**, and **`4`**:
  - `0`: No DR (Healthy)
  - `1`: Mild
  - `2`: Moderate
  - `3`: Severe
  - `4`: Proliferative DR
- Supported formats: **JPG, JPEG, PNG, BMP, TIFF**.
""")

col_zip_upload, col_train_action = st.columns([1, 1], gap="large")

with col_zip_upload:
    uploaded_zip = st.file_uploader(
        "Upload Labeled Dataset ZIP:",
        type=["zip"],
        help="Upload a ZIP file containing folders 0, 1, 2, 3, and 4."
    )

    use_demo_data = st.checkbox(
        "Use Educational Built-in Demo Dataset (20 synthetic fundus samples)",
        value=False,
        help="Check this to test the training workflow immediately without uploading an external ZIP."
    )

with col_train_action:
    st.markdown("#### Classifier Settings")
    test_ratio = st.slider("Hold-out Test Split Ratio:", min_value=0.15, max_value=0.40, value=0.25, step=0.05,
                           help="Proportion of the dataset reserved for testing the trained model.")
    n_trees = st.slider("Number of Random Forest Trees (n_estimators):", min_value=20, max_value=200, value=100, step=10,
                        help="More trees improve ensemble stability and classification robustness.")
    
    train_button = st.button("🚀 Process, Train & Evaluate Model", type="primary", use_container_width=True)

# Training logic execution
if train_button:
    zip_source = None
    if uploaded_zip is not None:
        zip_source = uploaded_zip
    elif use_demo_data:
        zip_source = io.BytesIO(sample_zip_bytes)
    else:
        st.error("❌ Please either upload a dataset ZIP archive or check 'Use Educational Built-in Demo Dataset'.")

    if zip_source is not None:
        with st.spinner("Validating dataset ZIP archive structure..."):
            is_valid, val_msg, class_files = validate_dataset_zip(zip_source)

        if not is_valid:
            st.error(f"❌ Dataset Validation Failed: {val_msg}")
        else:
            st.info(f"ℹ️ {val_msg}")
            
            # Progress bar for extraction & preprocessing
            prog_bar = st.progress(0.0)
            status_text = st.empty()

            def update_progress(current, total, info):
                pct = current / max(total, 1)
                prog_bar.progress(pct)
                status_text.text(f"Processing image {current} of {total}: {info}")

            try:
                # 1. Batch extract and process
                status_text.text("Extracting and preprocessing images from ZIP archive...")
                X, y, sample_names, error_log = extract_and_process_dataset(
                    zip_source, class_files, progress_callback=update_progress
                )
                
                if error_log:
                    with st.expander(f"⚠️ Skipped {len(error_log)} invalid file(s)"):
                        for err in error_log:
                            st.write(f"- {err}")

                # 2. Train model
                status_text.text("Training balanced Random Forest classifier...")
                model, eval_results = train_model(
                    X, y,
                    test_size=test_ratio,
                    random_state=42,
                    n_estimators=n_trees
                )

                prog_bar.progress(1.0)
                status_text.text("🎉 Model training and evaluation complete!")
                
                # Save to session state
                st.session_state.model = model
                st.session_state.eval_results = eval_results
                st.success("✅ Model successfully trained and evaluated on hold-out test set!")

            except Exception as e:
                st.error(f"❌ Error during dataset processing or training: {str(e)}")

st.divider()

# ==============================================================================
# SECTION 5: Results
# ==============================================================================
st.markdown("### 5. Results")
st.markdown("""
<div class="info-card">
    <strong>Why this step matters:</strong> We evaluate how well the trained Random Forest generalizes to hold-out test images using statistical metrics (Accuracy, Macro F1, Confusion Matrix), then inspect the model's prediction for the currently loaded fundus image.
</div>
""", unsafe_allow_html=True)

if st.session_state.eval_results is not None:
    res = st.session_state.eval_results

    # Summary metric cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{res['accuracy'] * 100:.1f}%</div>
            <div class="metric-label">Hold-out Test Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{res['macro_f1']:.3f}</div>
            <div class="metric-label">Macro F1-Score</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{res['n_train']}</div>
            <div class="metric-label">Training Samples</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{res['n_test']}</div>
            <div class="metric-label">Hold-out Test Samples</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.caption(f"ℹ️ **Split Protocol**: {res['stratify_note']}")

    col_cm, col_report = st.columns([1, 1], gap="large")

    with col_cm:
        st.markdown("#### Hold-out Confusion Matrix")
        st.caption("Rows show true clinical grades; columns show model predictions. Diagonal cells represent correct classifications.")
        fig_cm = plot_confusion_matrix(res['confusion_matrix'], res['classes_present'])
        st.pyplot(fig_cm, use_container_width=True)
        plt.close(fig_cm)

    with col_report:
        st.markdown("#### Per-Class Classification Report")
        st.caption("Precision measures false positive rate; Recall measures sensitivity; F1 balances both.")
        
        # Format classification report table
        report_data = []
        for c in res['classes_present']:
            c_key = f"{c} ({CLASS_LABELS[c]})"
            if c_key in res['report_dict']:
                stats = res['report_dict'][c_key]
                report_data.append({
                    "Grade": f"Class {c}",
                    "Clinical Stage": CLASS_LABELS[c],
                    "Precision": f"{stats['precision']:.2f}",
                    "Recall (Sensitivity)": f"{stats['recall']:.2f}",
                    "F1-Score": f"{stats['f1-score']:.2f}",
                    "Test Support": int(stats['support'])
                })
        
        if report_data:
            df_report = pd.DataFrame(report_data)
            st.dataframe(df_report, use_container_width=True, hide_index=True)

        st.markdown("#### Dataset Class Distribution")
        dist_data = []
        for c in res['classes_present']:
            dist_data.append({
                "Class": f"Grade {c} ({CLASS_LABELS[c]})",
                "Total Images": res['class_counts_total'].get(c, 0),
                "Train Split": res['train_counts'].get(c, 0),
                "Test Split": res['test_counts'].get(c, 0)
            })
        st.dataframe(pd.DataFrame(dist_data), use_container_width=True, hide_index=True)

    st.divider()

    # Inference on currently loaded image
    st.markdown("### 🔍 Model Prediction for Loaded Fundus Image")

    if st.session_state.features is not None and st.session_state.model is not None:
        pred_res = predict_single_image(
            st.session_state.model,
            st.session_state.features,
            res['classes_present']
        )

        pred_grade = pred_res['predicted_class']
        pred_name = pred_res['predicted_name']
        pred_score = pred_res['predicted_score']
        all_scores = pred_res['all_scores']

        col_pred_card, col_pred_chart = st.columns([1, 1], gap="large")

        with col_pred_card:
            badge_class = f"grade-badge-{pred_grade}"
            st.markdown(f"""
            <div class="step-box">
                <span class="{badge_class}">Predicted Severity: Grade {pred_grade}</span>
                <h3 style="margin-top: 0.5rem; color: #f8fafc;">{pred_name}</h3>
                <p style="color: #cbd5e1; font-size: 0.95rem;">{pred_res['predicted_description']}</p>
                <div style="margin-top: 1rem; padding: 0.75rem; background: #0f172a; border-radius: 8px;">
                    <div style="font-size: 0.85rem; color: #94a3b8;">MODEL PREDICTION SCORE</div>
                    <div style="font-size: 1.6rem; font-weight: 700; color: #38bdf8;">{pred_score * 100:.1f}%</div>
                    <div style="font-size: 0.8rem; color: #64748b; margin-top: 4px;">
                        * Proportion of decision trees voting for Grade {pred_grade}.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.warning("""
            **⚠️ Important Clinical Score Clarification:**
            The percentage score shown above is an ensemble decision-tree voting fraction, **NOT a calibrated clinical probability**.
            Machine-learning model consensus scores must never be treated as an absolute diagnostic probability.
            """)

        with col_pred_chart:
            st.markdown("#### Prediction Score Distribution Across Grades")
            
            # Bar chart of per-class probabilities
            chart_data = pd.DataFrame({
                "Grade": [f"Grade {c}\n{CLASS_LABELS[c]}" for c in range(5)],
                "Score": [all_scores[c] * 100 for c in range(5)]
            })

            fig_bar, ax_bar = plt.subplots(figsize=(6, 3.8), dpi=100)
            fig_bar.patch.set_facecolor("#0F172A")
            ax_bar.set_facecolor("#1E293B")
            
            bar_colors = ["#059669", "#0284c7", "#d97706", "#ea580c", "#dc2626"]
            bars = ax_bar.bar(chart_data["Grade"], chart_data["Score"], color=bar_colors, edgecolor="#334155")
            
            ax_bar.set_ylabel("Vote Score (%)", color="#94A3B8", fontsize=9)
            ax_bar.set_ylim(0, 105)
            ax_bar.tick_params(colors="#CBD5E1", labelsize=8)
            ax_bar.grid(axis="y", linestyle="--", alpha=0.2, color="#94A3B8")
            
            for bar in bars:
                height = bar.get_height()
                ax_bar.annotate(
                    f"{height:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    color="#F8FAFC", fontsize=8, fontweight="bold"
                )

            for spine in ax_bar.spines.values():
                spine.set_edgecolor("#334155")
            plt.tight_layout()
            st.pyplot(fig_bar, use_container_width=True)
            plt.close(fig_bar)

    elif st.session_state.features is None:
        st.info("👈 Please load and preprocess an image in Section 1 to view model inference.")
    else:
        st.info("Please train a model in Section 4 to generate predictions.")

else:
    st.info("👈 Train the model in Section 4 to view hold-out performance metrics, confusion matrix, and image prediction.")

st.markdown("---")
st.caption(
    "Retinal Disease Screening Lab • Educational Prototype • Built with Python, Streamlit, Scikit-learn, Scikit-image, and Pillow"
)
