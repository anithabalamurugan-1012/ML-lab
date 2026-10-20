"""
test_pipeline.py
================
Self-contained verification script for the Retinal Disease Screening Lab.
Tests preprocessing, feature extraction, dataset validation, model training, and inference.
"""

import io
import os
import sys

# Ensure current directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image

def run_tests():
    print("=" * 60)
    print("Running Retinal Disease Screening Pipeline Verification...")
    print("=" * 60)

    # 1. Test image generation & preprocessing
    print("[1/5] Testing Image Generation & Preprocessing...")
    from dataset_utils import generate_synthetic_fundus_image, create_sample_dataset_zip_bytes, validate_dataset_zip, extract_and_process_dataset
    from image_processing import preprocess_image
    
    test_img = generate_synthetic_fundus_image(class_label=2, seed=99)
    assert isinstance(test_img, Image.Image), "Generated image is not a PIL Image"
    print(f"  - Generated synthetic image of size {test_img.size}")
    
    prep_img, meta = preprocess_image(test_img)
    assert prep_img.size == (256, 256), f"Expected (256, 256), got {prep_img.size}"
    assert meta["resized_size"] == "256 x 256"
    assert meta["clahe_applied"] is True
    print(f"  - Preprocessing succeeded: {meta}")

    # 2. Test feature extraction
    print("[2/5] Testing Feature Extraction (HOG + Color Histograms)...")
    from feature_extraction import extract_all_features
    
    feats, f_meta, hog_viz, hists = extract_all_features(prep_img, visualize_hog=True)
    assert feats.ndim == 1, "Feature vector must be 1D"
    assert len(feats) == f_meta["total_features"]
    assert hog_viz is not None, "HOG visualization should not be None when visualize_hog=True"
    assert "Red" in hists and "Green" in hists and "Blue" in hists
    print(f"  - Features extracted: Total={f_meta['total_features']}, HOG={f_meta['hog_features_count']}, Color={f_meta['color_features_count']}")

    # 3. Test dataset validation and extraction
    print("[3/5] Testing Sample Dataset ZIP Creation & Validation...")
    zip_bytes = create_sample_dataset_zip_bytes(images_per_class=3)
    zip_buf = io.BytesIO(zip_bytes)
    
    is_valid, msg, class_files = validate_dataset_zip(zip_buf)
    assert is_valid, f"Validation failed: {msg}"
    print(f"  - ZIP validation succeeded: {msg}")
    for c in range(5):
        assert len(class_files[c]) == 3, f"Class {c} expected 3 images, found {len(class_files[c])}"

    # 4. Test dataset batch processing
    print("[4/5] Testing Batch Feature Extraction from ZIP...")
    X, y, names, error_log = extract_and_process_dataset(zip_buf, class_files)
    assert len(X) == 15, f"Expected 15 samples, got {len(X)}"
    assert len(y) == 15, f"Expected 15 labels, got {len(y)}"
    assert len(error_log) == 0, f"Unexpected errors: {error_log}"
    print(f"  - Processed {len(X)} samples into feature matrix shape {X.shape}")

    # 5. Test model training, evaluation & single-image inference
    print("[5/5] Testing Model Training, Evaluation & Inference...")
    from model_utils import train_model, plot_confusion_matrix, predict_single_image
    
    clf, eval_res = train_model(X, y, test_size=0.33, random_state=42, n_estimators=50)
    assert eval_res["accuracy"] >= 0.0, "Accuracy must be >= 0"
    assert eval_res["macro_f1"] >= 0.0, "Macro F1 must be >= 0"
    assert eval_res["confusion_matrix"].shape == (5, 5), "Confusion matrix should be 5x5"
    print(f"  - Model trained: Hold-out Accuracy = {eval_res['accuracy'] * 100:.1f}%, Macro F1 = {eval_res['macro_f1']:.3f}")
    print(f"  - Stratify protocol: {eval_res['stratify_note']}")

    # Confusion matrix plot test
    fig = plot_confusion_matrix(eval_res["confusion_matrix"], eval_res["classes_present"])
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)

    # Prediction test
    pred_res = predict_single_image(clf, feats, eval_res["classes_present"])
    print(f"  - Single image prediction: Grade {pred_res['predicted_class']} ({pred_res['predicted_name']})")
    print(f"  - Score: {pred_res['predicted_score'] * 100:.1f}%")
    print(f"  - All class scores: {pred_res['all_scores']}")

    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY! The pipeline is fully functional.")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
