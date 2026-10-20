"""
dataset_utils.py
================
Utilities for inspecting, validating, and extracting labeled diabetic
retinopathy dataset ZIP archives, plus synthetic sample dataset generation
for educational testing.

Expected ZIP directory structure:
  dataset.zip
  ├── 0/   (No DR)
  ├── 1/   (Mild)
  ├── 2/   (Moderate)
  ├── 3/   (Severe)
  └── 4/   (Proliferative DR)
"""

# pylint: disable=too-many-locals,too-many-branches
# pylint: disable=too-many-statements,too-many-arguments
# pylint: disable=too-many-positional-arguments

import io
import os
import zipfile
from collections.abc import Callable
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from feature_extraction import extract_all_features
from image_processing import preprocess_image

CLASS_LABELS: dict[int, str] = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}

CLASS_NAME_ALIASES: dict[str, int] = {
    # Class 0: No DR / Normal
    "0": 0,
    "no dr": 0,
    "no_dr": 0,
    "nodr": 0,
    "normal": 0,
    "healthy": 0,
    "class 0": 0,
    "class_0": 0,
    "grade 0": 0,
    "grade_0": 0,
    # Class 1: Mild
    "1": 1,
    "mild": 1,
    "mild dr": 1,
    "mild_dr": 1,
    "class 1": 1,
    "class_1": 1,
    "grade 1": 1,
    "grade_1": 1,
    # Class 2: Moderate
    "2": 2,
    "moderate": 2,
    "moderate dr": 2,
    "moderate_dr": 2,
    "class 2": 2,
    "class_2": 2,
    "grade 2": 2,
    "grade_2": 2,
    # Class 3: Severe
    "3": 3,
    "severe": 3,
    "severe dr": 3,
    "severe_dr": 3,
    "class 3": 3,
    "class_3": 3,
    "grade 3": 3,
    "grade_3": 3,
    # Class 4: Proliferative
    "4": 4,
    "proliferative": 4,
    "proliferative dr": 4,
    "proliferative_dr": 4,
    "pdr": 4,
    "class 4": 4,
    "class_4": 4,
    "grade 4": 4,
    "grade_4": 4,
}

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _match_folder_class(parts: list[str]) -> int | None:
    """Matches class id from enclosing folder parts (innermost first)."""
    for folder in reversed(parts[:-1]):
        folder_lower = folder.lower()
        if folder_lower in CLASS_NAME_ALIASES:
            return CLASS_NAME_ALIASES[folder_lower]
        if (
            folder
            and folder[0] in {"0", "1", "2", "3", "4"}
            and (len(folder) == 1 or not folder[1].isdigit())
        ):
            return int(folder[0])
    return None


def _scan_zip_for_classes(z: zipfile.ZipFile) -> dict[int, list[str]]:
    """Scans all entries in the open ZIP archive and groups by DR class."""
    class_files: dict[int, list[str]] = {c: [] for c in range(5)}
    for path in z.namelist():
        clean_path = path.replace("\\", "/")
        parts = [p.strip() for p in clean_path.split("/") if p.strip()]

        if (
            len(parts) < 2
            or "__MACOSX" in parts
            or clean_path.endswith("/")
        ):
            continue

        basename = parts[-1]
        if basename.startswith(".") or basename.lower() == "thumbs.db":
            continue

        ext = os.path.splitext(basename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            continue

        matched_class = _match_folder_class(parts)
        if matched_class is not None:
            class_files[matched_class].append(path)

    return class_files


def validate_dataset_zip(
    zip_file: Any,
) -> tuple[bool, str, dict[int, list[str]]]:
    """
    Validates a dataset ZIP file for expected class folders 0, 1, 2, 3, 4.

    Parameters:
        zip_file: Uploaded file object, file path, or bytes buffer.

    Returns:
        Tuple:
        - is_valid (bool): True if valid classes and images are found.
        - message (str): Informational or error message explaining result.
        - class_files (dict): Map of class_id (0-4) -> list of file paths.
    """
    try:
        if isinstance(zip_file, bytes):
            zip_file = io.BytesIO(zip_file)
        elif hasattr(zip_file, "seek"):
            zip_file.seek(0)

        if not zipfile.is_zipfile(zip_file):
            return False, "The uploaded file is not a valid ZIP archive.", {}

        if hasattr(zip_file, "seek"):
            zip_file.seek(0)
    except (OSError, ValueError, zipfile.BadZipFile) as err:
        return False, f"Failed to inspect ZIP file: {err!s}", {}

    try:
        with zipfile.ZipFile(zip_file, "r") as z:
            class_files = _scan_zip_for_classes(z)
    except (OSError, ValueError, zipfile.BadZipFile, KeyError) as err:
        return (
            False,
            f"Error reading entries inside the ZIP archive: {err!s}",
            {},
        )

    total_images = sum(len(files) for files in class_files.values())
    if total_images == 0:
        return (
            False,
            (
                "No supported images (.jpg, .jpeg, .png, .bmp, .tiff) were "
                "found inside class folders named '0', '1', '2', '3', or '4'. "
                "Please ensure your ZIP has folders named 0 through 4 "
                "containing retinal fundus images."
            ),
            class_files,
        )

    empty_classes = [c for c, files in class_files.items() if len(files) == 0]
    classes_with_few = [
        c for c, files in class_files.items() if 0 < len(files) < 2
    ]

    status_notes = []
    if empty_classes:
        empty_names = [f"Class {c} ({CLASS_LABELS[c]})" for c in empty_classes]
        status_notes.append(f"Missing images in: {', '.join(empty_names)}.")

    if classes_with_few:
        few_names = [
            f"Class {c} ({len(class_files[c])} image)"
            for c in classes_with_few
        ]
        status_notes.append(
            f"Classes with < 2 samples: {', '.join(few_names)} "
            "(requires >=2 for stratified train/test split)."
        )

    active_classes = [c for c, files in class_files.items() if len(files) > 0]
    if len(active_classes) < 2:
        return (
            False,
            (
                "Classification requires at least 2 distinct classes. "
                f"Only class {active_classes[0]} was found with "
                f"{len(class_files[active_classes[0]])} images."
            ),
            class_files,
        )

    summary_msg = (
        f"Found {total_images} total images across "
        f"{len(active_classes)} classes."
    )
    if status_notes:
        summary_msg += " Note: " + " ".join(status_notes)

    return True, summary_msg, class_files


def extract_and_process_dataset(
    zip_file: Any,
    class_files: dict[int, list[str]],
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Extracts images from ZIP archive, runs preprocessing and features,
    and returns feature matrix X and label vector y.

    Parameters:
        zip_file: Uploaded file or path to ZIP archive.
        class_files: Dict mapping class_id to file paths inside ZIP.
        progress_callback: Optional callback receiving (done, total, info).

    Returns:
        Tuple:
        - X (np.ndarray): 2D feature matrix (N_samples, N_features).
        - y (np.ndarray): 1D integer label array (N_samples,).
        - sample_names (list[str]): List of processed image filenames.
        - error_log (list[str]): Log of skipped/corrupt files.
    """
    items_to_process: list[tuple[int, str]] = []
    for c_id, paths in sorted(class_files.items()):
        for p in paths:
            items_to_process.append((c_id, p))

    total = len(items_to_process)
    x_list: list[np.ndarray] = []
    y_list: list[int] = []
    sample_names: list[str] = []
    error_log: list[str] = []

    if isinstance(zip_file, bytes):
        zip_file = io.BytesIO(zip_file)
    elif hasattr(zip_file, "seek"):
        zip_file.seek(0)

    with zipfile.ZipFile(zip_file, "r") as z:
        for idx, (class_id, internal_path) in enumerate(items_to_process):
            filename = os.path.basename(internal_path)
            if progress_callback:
                progress_callback(
                    idx + 1,
                    total,
                    f"Class {class_id} ({CLASS_LABELS[class_id]}): {filename}",
                )

            try:
                img_data = z.read(internal_path)
                image_buf = io.BytesIO(img_data)
                with Image.open(image_buf) as img_raw:
                    image = img_raw.convert("RGB")

                prep_img, _ = preprocess_image(image)
                features, _, _, _ = extract_all_features(
                    prep_img, visualize_hog=False
                )

                x_list.append(features)
                y_list.append(class_id)
                sample_names.append(f"{class_id}_{filename}")
            except (OSError, ValueError, zipfile.BadZipFile) as err:
                error_log.append(
                    f"Failed to process '{internal_path}': {err!s}"
                )

    if not x_list:
        raise ValueError(
            "No valid images could be processed from the archive."
        )

    x_matrix = np.array(x_list, dtype=np.float32)
    y_array = np.array(y_list, dtype=np.int64)
    return x_matrix, y_array, sample_names, error_log


def _get_lesion_counts(
    class_label: int, rng: np.random.Generator
) -> tuple[int, int, int, int, bool]:
    """Generates counts for lesions (MA, exudates, hems, CWS, neovasc)."""
    if class_label == 1:
        return int(rng.integers(3, 7)), 0, 0, 0, False
    if class_label == 2:
        return (
            int(rng.integers(8, 16)),
            int(rng.integers(5, 12)),
            0,
            0,
            False,
        )
    if class_label == 3:
        return (
            int(rng.integers(15, 25)),
            int(rng.integers(12, 22)),
            int(rng.integers(6, 12)),
            int(rng.integers(2, 5)),
            False,
        )
    if class_label == 4:
        return (
            int(rng.integers(20, 35)),
            int(rng.integers(18, 30)),
            int(rng.integers(15, 25)),
            int(rng.integers(4, 8)),
            True,
        )
    return 0, 0, 0, 0, False


def _draw_optic_disc_and_vessels(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    radius: int,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Draws optic disc and major retinal vascular arcades."""
    disc_x = center[0] - int(radius * 0.42)
    disc_y = center[1]
    disc_rx, disc_ry = 18, 22
    draw.ellipse(
        [
            disc_x - disc_rx,
            disc_y - disc_ry,
            disc_x + disc_rx,
            disc_y + disc_ry,
        ],
        fill=(245, 220, 130),
    )

    vessel_color = (110, 20, 15)
    for branch in [(-0.8, -1.0), (-0.8, 1.0), (1.0, -0.6), (1.0, 0.6)]:
        pts = [(disc_x, disc_y)]
        curr_x, curr_y = float(disc_x), float(disc_y)
        for _ in range(5):
            curr_x += branch[0] * rng.uniform(15, 25) + rng.uniform(-4, 4)
            curr_y += branch[1] * rng.uniform(15, 25) + rng.uniform(-4, 4)
            pts.append((int(curr_x), int(curr_y)))
        if len(pts) > 1:
            draw.line(pts, fill=vessel_color, width=3)

    return disc_x, disc_y


def _draw_lesions(
    draw: ImageDraw.ImageDraw,
    geometry: tuple[tuple[int, int], int],
    rng: np.random.Generator,
    disc_pos: tuple[int, int],
    counts: tuple[int, int, int, int, bool],
) -> None:
    """Draws DR lesions: MA, exudates, hemorrhages, CWS, neovasc."""
    center, radius = geometry
    num_ma, num_exudates, num_hems, num_cws, neovasc = counts
    disc_x, disc_y = disc_pos

    # Microaneurysms (small dark red dots)
    for _ in range(num_ma):
        angle = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(25, radius - 20)
        x = int(center[0] + r * np.cos(angle))
        y = int(center[1] + r * np.sin(angle))
        draw.ellipse([x - 1, y - 1, x + 2, y + 2], fill=(90, 5, 5))

    # Hard Exudates (bright yellow lipid deposits)
    for _ in range(num_exudates):
        angle = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(30, radius - 25)
        x = int(center[0] + r * np.cos(angle))
        y = int(center[1] + r * np.sin(angle))
        draw.ellipse([x - 2, y - 2, x + 3, y + 3], fill=(255, 245, 160))

    # Hemorrhages (dark red blotches)
    for _ in range(num_hems):
        angle = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(35, radius - 30)
        x = int(center[0] + r * np.cos(angle))
        y = int(center[1] + r * np.sin(angle))
        hr = int(rng.integers(3, 7))
        draw.ellipse([x - hr, y - hr, x + hr, y + hr], fill=(80, 0, 0))

    # Cotton Wool Spots (pale fluffy white/gray patches)
    for _ in range(num_cws):
        angle = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(30, radius - 35)
        x = int(center[0] + r * np.cos(angle))
        y = int(center[1] + r * np.sin(angle))
        cwr = int(rng.integers(5, 9))
        draw.ellipse(
            [x - cwr, y - cwr, x + cwr, y + cwr], fill=(225, 225, 230)
        )

    # Neovascularization (chaotic microvascular loops)
    if neovasc:
        for _ in range(6):
            start_x = int(disc_x + rng.integers(-20, 20))
            start_y = int(disc_y + rng.integers(-20, 20))
            neo_pts = [(start_x, start_y)]
            for _ in range(4):
                next_x = int(neo_pts[-1][0] + int(rng.integers(-10, 10)))
                next_y = int(neo_pts[-1][1] + int(rng.integers(-10, 10)))
                neo_pts.append((next_x, next_y))
            draw.line(neo_pts, fill=(140, 10, 10), width=1)


def generate_synthetic_fundus_image(
    class_label: int, seed: int = 42
) -> Image.Image:
    """
    Generates an educational synthetic retinal fundus-like image.
    Produces landmarks (dark background, orange/red retina, optic disc,
    vascular arcades) along with severity-specific lesions:
      0 (No DR): Clean retina and clear vessels.
      1 (Mild): A few microaneurysms (tiny dark-red dots).
      2 (Moderate): Microaneurysms and hard exudates (bright yellowish specks).
      3 (Severe): Multiple hemorrhages, hard exudates, cotton-wool spots.
      4 (Proliferative DR): Hemorrhages, neovascularization.
    """
    rng = np.random.default_rng(seed)
    size = (300, 300)
    img = Image.new("RGB", size, (12, 10, 10))
    draw = ImageDraw.Draw(img)

    center = (150, 150)
    radius = 130

    retina_color = (
        int(rng.integers(180, 215)),
        int(rng.integers(70, 95)),
        int(rng.integers(30, 50)),
    )
    draw.ellipse(
        [
            center[0] - radius,
            center[1] - radius,
            center[0] + radius,
            center[1] + radius,
        ],
        fill=retina_color,
    )

    disc_pos = _draw_optic_disc_and_vessels(draw, center, radius, rng)
    counts = _get_lesion_counts(class_label, rng)
    _draw_lesions(draw, (center, radius), rng, disc_pos, counts)

    return img


def create_sample_dataset_zip_bytes(images_per_class: int = 4) -> bytes:
    """
    Creates an in-memory ZIP archive containing an educational demo dataset
    with `images_per_class` synthetic fundus images for each of the 5 classes.
    """
    with io.BytesIO() as buf:
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for c in range(5):
                for i in range(images_per_class):
                    seed = 1000 + c * 100 + i
                    with generate_synthetic_fundus_image(
                        class_label=c, seed=seed
                    ) as img, io.BytesIO() as img_bytes:
                        img.save(img_bytes, format="JPEG", quality=90)
                        z.writestr(
                            f"{c}/sample_class{c}_{i + 1}.jpg",
                            img_bytes.getvalue(),
                        )
        return buf.getvalue()
