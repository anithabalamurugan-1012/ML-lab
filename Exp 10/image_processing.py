"""
image_processing.py
===================
Image preprocessing pipeline for the Retinal Disease Screening Lab.

Key preprocessing steps for retinal fundus images:
1. EXIF Orientation Correction: Ensures fundus images are correctly aligned.
2. Dark Border Cropping: Removes non-informative black circular borders.
3. Resizing: Standardizes image dimensions to 256x256 pixels.
4. CLAHE (Contrast Limited Adaptive Histogram Equalization): Enhances local
   contrast to highlight retinal lesions (microaneurysms, hemorrhages, exudates).
"""

from typing import Tuple, Dict, Any, Union
from PIL import Image, ImageOps
import numpy as np
from skimage import exposure


def correct_exif_orientation(image: Image.Image) -> Tuple[Image.Image, bool]:
    """
    Corrects image orientation according to EXIF metadata if present.
    
    Parameters:
        image: PIL Image object.
        
    Returns:
        Tuple of (oriented PIL Image, boolean indicating if transposition was applied).
    """
    try:
        transposed = ImageOps.exif_transpose(image)
        if transposed is not None:
            was_transposed = transposed.size != image.size
            return transposed, was_transposed
    except Exception:
        pass
    return image, False


def crop_dark_borders(image: Image.Image, tol: int = 15) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """
    Crops uninformative dark borders surrounding circular fundus photographs.
    
    Parameters:
        image: PIL Image object (RGB).
        tol: Intensity threshold below which pixels are considered dark background.
        
    Returns:
        Tuple of (cropped PIL Image, bounding box tuple (xmin, ymin, xmax, ymax)).
    """
    img_gray = np.array(image.convert("L"))
    mask = img_gray > tol
    
    # If the mask is mostly empty (e.g., extremely dark image), don't crop
    if not np.any(mask):
        return image, (0, 0, image.width, image.height)
    
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    
    # Add a tiny 2-pixel margin if possible to avoid cutting retina edge
    ymin = max(0, int(ymin) - 2)
    xmin = max(0, int(xmin) - 2)
    ymax = min(image.height, int(ymax) + 3)
    xmax = min(image.width, int(xmax) + 3)
    
    cropped = image.crop((xmin, ymin, xmax, ymax))
    return cropped, (xmin, ymin, xmax, ymax)


def resize_image(image: Image.Image, target_size: Tuple[int, int] = (256, 256)) -> Image.Image:
    """
    Resizes image to standard resolution for feature extraction.
    
    Parameters:
        image: PIL Image object.
        target_size: Desired (width, height) tuple, default (256, 256).
        
    Returns:
        Resized PIL Image.
    """
    # Use LANCZOS (or BICUBIC) resampling for high quality anti-aliasing
    return image.resize(target_size, resample=Image.Resampling.LANCZOS)


def apply_clahe(image: Image.Image, clip_limit: float = 0.03) -> Image.Image:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
    CLAHE compensates for non-uniform illumination across the retina,
    sharpening vascular structure and subtle diabetic retinopathy lesions.
    
    Parameters:
        image: PIL Image object in RGB mode.
        clip_limit: Normalized clipping limit for contrast enhancement [0.01 - 0.05].
        
    Returns:
        Enhanced PIL Image in RGB mode.
    """
    # Convert PIL Image to float numpy array in range [0, 1]
    img_arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    
    # Apply CLAHE across color channels
    # skimage's equalize_adapthist supports 3D arrays (H, W, C)
    clahe_arr = exposure.equalize_adapthist(img_arr, clip_limit=clip_limit)
    
    # Scale back to uint8 [0, 255]
    clahe_uint8 = np.clip(clahe_arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(clahe_uint8, mode="RGB")


def preprocess_image(image_input: Union[Image.Image, str]) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Full preprocessing pipeline:
    1. EXIF orientation correction
    2. Convert to RGB mode
    3. Dark border cropping
    4. Resizing to 256x256
    5. CLAHE contrast enhancement
    
    Parameters:
        image_input: PIL Image or path to image.
        
    Returns:
        Tuple of (preprocessed PIL Image, dict of metadata describing transformations).
    """
    image: Image.Image
    if isinstance(image_input, str):
        image = Image.open(image_input)
    else:
        image = image_input
        
    metadata: Dict[str, Any] = {}
    original_size = image.size
    metadata["original_size"] = f"{original_size[0]} x {original_size[1]}"
    
    # 1. EXIF orientation
    oriented_image, exif_applied = correct_exif_orientation(image)
    metadata["exif_corrected"] = exif_applied
    
    # Ensure RGB
    rgb_image = oriented_image.convert("RGB")
    
    # 2. Crop dark borders
    cropped_image, bbox = crop_dark_borders(rgb_image, tol=15)
    metadata["crop_bbox"] = bbox
    metadata["cropped_size"] = f"{cropped_image.width} x {cropped_image.height}"
    
    # 3. Resize to 256x256
    resized_image = resize_image(cropped_image, target_size=(256, 256))
    metadata["resized_size"] = "256 x 256"
    
    # 4. CLAHE contrast enhancement
    clahe_image = apply_clahe(resized_image, clip_limit=0.03)
    metadata["clahe_applied"] = True
    metadata["clahe_clip_limit"] = 0.03
    
    return clahe_image, metadata
