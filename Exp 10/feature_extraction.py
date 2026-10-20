"""
feature_extraction.py
=====================
Feature extraction module for diabetic retinopathy classification:
1. HOG (Histogram of Oriented Gradients):
   Captures retinal blood vessel structure, disc edges, microvascular abnormalities,
   and lesion texture.
2. RGB Color Histograms:
   Captures retinal pigment distribution, optic disc brightness, dark hemorrhages,
   and bright lipid exudates.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray


def extract_hog_features(
    image: Image.Image,
    orientations: int = 8,
    pixels_per_cell: Tuple[int, int] = (32, 32),
    cells_per_block: Tuple[int, int] = (2, 2),
    visualize: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Extracts Histogram of Oriented Gradients (HOG) features from a 256x256 fundus image.
    
    Parameters:
        image: PIL Image in RGB format (assumed preprocessed to 256x256).
        orientations: Number of gradient orientation bins.
        pixels_per_cell: Size (in pixels) of a HOG cell.
        cells_per_block: Number of cells in each block for L2 normalization.
        visualize: If True, also returns a 2D visualization array of the HOG gradients.
        
    Returns:
        Tuple of (1D numpy feature vector, optional 2D HOG visualization array).
    """
    # Convert image to numpy array
    img_np = np.array(image.convert("RGB"))
    
    # Retinal structures and lesions contrast strongly in green/luminance channel
    # Convert to grayscale for gradient orientation extraction
    img_gray = rgb2gray(img_np)
    
    if visualize:
        features, hog_image = hog(
            img_gray,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            visualize=True,
            feature_vector=True,
            channel_axis=None
        )
        return features.astype(np.float32), hog_image
    else:
        features = hog(
            img_gray,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            visualize=False,
            feature_vector=True,
            channel_axis=None
        )
        return features.astype(np.float32), None


def extract_color_histograms(
    image: Image.Image,
    bins: int = 32
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Extracts normalized RGB color histograms from the fundus image.
    
    Parameters:
        image: PIL Image in RGB format.
        bins: Number of bins per color channel (default 32).
        
    Returns:
        Tuple of (flattened 1D numpy array of concatenated histograms, dict of per-channel hists).
    """
    img_np = np.array(image.convert("RGB"))
    
    # Calculate histogram for each channel (R, G, B)
    channel_hists = {}
    hist_list = []
    
    for idx, color_name in enumerate(["Red", "Green", "Blue"]):
        channel_data = img_np[:, :, idx].ravel()
        # Compute histogram with bins across [0, 256]
        hist, bin_edges = np.histogram(channel_data, bins=bins, range=(0, 256))
        
        # Normalize histogram so sum equals 1.0 (invariant to image size)
        hist_norm = hist.astype(np.float32) / (channel_data.size + 1e-7)
        channel_hists[color_name] = hist_norm
        hist_list.append(hist_norm)
        
    combined_hist = np.concatenate(hist_list)
    return combined_hist, channel_hists


def extract_all_features(
    image: Image.Image,
    visualize_hog: bool = False
) -> Tuple[np.ndarray, Dict[str, Any], Optional[np.ndarray], Dict[str, np.ndarray]]:
    """
    Combines HOG texture features and RGB color histogram features into a single feature vector.
    
    Parameters:
        image: Preprocessed PIL Image (256x256, RGB).
        visualize_hog: Whether to generate the HOG gradient visualization image.
        
    Returns:
        Tuple containing:
        - feature_vector: 1D concatenated numpy array
        - metadata: Dictionary detailing feature breakdown and counts
        - hog_image: Optional 2D HOG visualization array
        - channel_hists: Dictionary containing per-channel color histograms
    """
    hog_feats, hog_img = extract_hog_features(image, visualize=visualize_hog)
    color_feats, channel_hists = extract_color_histograms(image, bins=32)
    
    full_vector = np.concatenate([hog_feats, color_feats])
    
    metadata = {
        "total_features": len(full_vector),
        "hog_features_count": len(hog_feats),
        "color_features_count": len(color_feats),
        "hog_orientations": 8,
        "hog_cell_size": "32 x 32",
        "hog_block_size": "2 x 2 cells",
        "color_bins_per_channel": 32,
        "channels": ["Red", "Green", "Blue"]
    }
    
    return full_vector, metadata, hog_img, channel_hists
