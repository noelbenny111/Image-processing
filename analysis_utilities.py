"""
analysis_utils.py

Utility functions for CZI image processing and blob detection.
Contains helpers for file I/O, image preprocessing, blob detection,
and the interactive GUI.
"""

import os
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Ellipse, Circle
import scipy.stats as stats
from aicspylibczi import CziFile
import czifile

# ------------------ Measurement & File Helpers ------------------

def calculate_measurement_errors(diameters: List[float], reference_diameter: Optional[float] = None) -> Dict[str, Any]:
    """Calculate various error metrics for diameter measurements."""
    if not diameters:
        return {
            "mean": 0, "std": 0, "cv_percent": 0, "error_percent": 0,
            "confidence_interval": (0, 0)
        }

    mean_diam = float(np.mean(diameters))
    std_diam = float(np.std(diameters))

    # Coefficient of variation (relative standard deviation)
    cv_percent = (std_diam / mean_diam) * 100 if mean_diam > 0 else 0

    # Percentage error compared to reference (if provided)
    error_percent = 0.0
    if reference_diameter is not None and reference_diameter > 0:
        error_percent = ((mean_diam - reference_diameter) / reference_diameter) * 100

    # 95% confidence interval
    if len(diameters) > 1:
        ci = stats.t.interval(0.95, len(diameters) - 1, loc=mean_diam, scale=stats.sem(diameters))
    else:
        ci = (mean_diam, mean_diam)

    return {
        "mean": mean_diam,
        "std": std_diam,
        "cv_percent": cv_percent,
        "error_percent": error_percent,
        "confidence_interval": ci
    }


def extract_degree_columns(img: np.ndarray, deg_start: int, deg_end: int, column_width: int) -> Tuple[Dict[int, Dict[str, Any]], np.ndarray]:
    """Extract vertical columns for each degree from a single 5x image."""
    all_degrees = list(range(deg_start, deg_end - 1, -1))
    regions = {}
    h, w = img.shape
    
    for idx, deg in enumerate(all_degrees):
        x_min = idx * column_width
        x_max = (idx + 1) * column_width

        if x_min >= w:
            print(f"Warning: Not enough width for degree {deg}. Image width: {w}, required: {x_min}")
            continue

        x_max = min(x_max, w)  # Ensure x_max does not exceed image width

        regions[deg] = {
            'strip': img[:, x_min:x_max],
            'x_min': x_min,
            'x_max': x_max,
            'x_center': (x_min + x_max) / 2
        }

    proj = np.nan_to_num(img, nan=0.0).sum(axis=0)
    proj_norm = (proj - proj.min())
    if proj_norm.max() > 0:
        proj_norm /= proj_norm.max()

    return regions, proj_norm


def get_pixel_size_um_from_czi(czi_path: str) -> Optional[float]:
    """Extract X pixel size in micrometers from CZI metadata using aicspylibczi."""
    try:
        czi = CziFile(czi_path)
        meta = czi.meta
        for elem in meta.iter():
            if elem.tag.endswith("Distance") and elem.attrib.get("Id", "").upper() == "X":
                val = elem.find("./Value")
                if val is not None and val.text:
                    return float(val.text) * 1e6  # Convert meters to micrometers
    except Exception as e:
        print(f"Warning: Failed to extract pixel size from CZI metadata for {czi_path}. Error: {e}")
    return None


def extract_degree_from_filename(filename: str) -> Optional[int]:
    """Extract degree from filename (first two characters)."""
    basename = os.path.basename(filename)
    name_without_ext = os.path.splitext(basename)[0]
    degree_str = name_without_ext[:2]
    try:
        return int(degree_str)
    except ValueError:
        print(f"Warning: Could not extract degree from filename: {filename}")
        return None


def load_czi_image(czi_path: str) -> np.ndarray:
    """Load a CZI file and return a single 2D float32 grayscale plane."""
    arr = czifile.imread(czi_path)
    arr = np.squeeze(arr)

    img: np.ndarray
    if arr.ndim == 2:
        img = arr
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        # Convert RGB to grayscale
        rgb_weights = np.array([0.2989, 0.5870, 0.1140])
        img = np.dot(arr[..., :3], rgb_weights)
    elif arr.ndim == 3:
        # Default to first channel if not RGB
        img = arr[..., 0]
    else:
        raise ValueError(f"Unsupported array shape after squeeze: {arr.shape} from {czi_path}")

    return img.astype(np.float32)


def crop_image(img: np.ndarray, x_start: int, width: int) -> np.ndarray:
    """Crop the image to a specified width starting from x_start."""
    return img[:, x_start : x_start + width]


# ------------------ Image Preprocessing ------------------

def preprocess_image_for_laser_spots(img: np.ndarray, clahe_clip: float = 2.0, tophat_kernel: int = 15, median_k: int = 3) -> np.ndarray:
    """Preprocessing for laser spots (bright spots on dark background)."""
    img_uint8 = (img / img.max() * 255).astype(np.uint8) if img.dtype != np.uint8 else img
    
    # NOTE: The original code inverted the image here, which is unusual for
    # bright spots on a dark background. This is kept to match behavior.
    # If spots are bright, you might want to remove this line.
    img_inverted = cv2.bitwise_not(img_uint8)
    
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(50, 50))
    img_clahe = clahe.apply(img_inverted)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tophat_kernel, tophat_kernel))
    img_tophat = cv2.morphologyEx(img_clahe, cv2.MORPH_BLACKHAT, kernel)

    img_blur = cv2.medianBlur(img_tophat, median_k)
    return img_blur


def preprocess_image_for_dark_spots(img: np.ndarray, clahe_clip: float = 2.0, tophat_kernel: int = 15, median_k: int = 3, sigma: float = 1.0, scratch_removal: bool = True) -> np.ndarray:
    """Preprocessing for dark spots on white background with scratch removal."""
    img_uint8 = (img / img.max() * 255).astype(np.uint8) if img.dtype != np.uint8 else img
    img_inverted = cv2.bitwise_not(img_uint8)
    img_blurred = cv2.GaussianBlur(img_inverted, (0, 0), sigmaX=sigma, sigmaY=sigma)

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(3, 3))
    img_clahe = clahe.apply(img_blurred)

    if scratch_removal:
        scratch_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (tophat_kernel, 3))
        img_cleaned = cv2.morphologyEx(img_clahe, cv2.MORPH_OPEN, scratch_kernel)
    else:
        img_cleaned = img_clahe

    spot_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tophat_kernel, tophat_kernel))
    img_tophat = cv2.morphologyEx(img_cleaned, cv2.MORPH_TOPHAT, spot_kernel)

    img_final = cv2.medianBlur(img_tophat, median_k)
    return img_final

# ------------------ Blob Detection & GUI ------------------

def detect_blobs_in_processed_image(
    processed_img: np.ndarray, 
    thresh_val: int = 128, 
    morph_iter: int = 2, 
    min_diam: float = 5, 
    max_diam: float = 250,
    min_contour_area: int = 20,
    invert_threshold: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Run morphological + threshold + contour detection on a preprocessed image."""
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph = cv2.morphologyEx(processed_img, cv2.MORPH_CLOSE, kernel, iterations=morph_iter)

    _, thresh = cv2.threshold(morph, thresh_val, 255, cv2.THRESH_BINARY)

    if invert_threshold:
        thresh = cv2.bitwise_not(thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_contour_area:
            continue
        
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (xc, yc), (MA, ma), angle = ellipse
            avg_diam = np.sqrt(MA * ma)
        else:
            (xc, yc), radius = cv2.minEnclosingCircle(cnt)
            ellipse = None
            avg_diam = radius * 2.0
            
        if min_diam <= avg_diam <= max_diam:
            blobs.append({
                "cnt": cnt, 
                "diam": avg_diam, 
                "center": (float(xc), float(yc)), 
                "ellipse": ellipse
            })
    return processed_img, thresh, blobs


def manual_deselect(img_uint8: np.ndarray, blobs: List[Dict[str, Any]], title: str = "Manual Selection") -> List[Dict[str, Any]]:
    """Show blobs overlay and let user click to toggle selection."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img_uint8, cmap='gray')

    patches = []
    selected = [True] * len(blobs)
    
    instruction = "Click blobs to toggle. 'a'=select all, 'd'=deselect all, 'enter'=finish."
    ax.text(0.5, 0.02, instruction, transform=ax.transAxes, ha='center',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))

    for i, b in enumerate(blobs):
        if b.get("ellipse") is not None:
            (xc, yc), (MA, ma), angle = b["ellipse"]
            p = Ellipse((xc, yc), MA, ma, angle=angle, edgecolor='lime', facecolor='none', lw=1, alpha=0.8)
            ax.text(xc, yc, str(i), color='lime', fontsize=10, ha='center', va='center', weight='bold')
        else:
            radius = b.get("diam", 10) / 2
            p = Circle(b["center"], radius=radius, edgecolor='lime', facecolor='none', lw=1, alpha=0.8)
            ax.text(b["center"][0], b["center"][1], str(i), color='lime', fontsize=10, ha='center', va='center', weight='bold')
        
        ax.add_patch(p)
        patches.append(p)

    ax.set_title(title)
    ax.axis('off')

    def update_display():
        for i, p in enumerate(patches):
            p.set_edgecolor('lime' if selected[i] else 'red')
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax: return
        for i, p in enumerate(patches):
            contains, _ = p.contains(event)
            if contains:
                selected[i] = not selected[i]
                update_display()
                break

    def on_key(event):
        if event.key == 'a':
            selected[:] = [True] * len(selected)
        elif event.key == 'd':
            selected[:] = [False] * len(selected)
        elif event.key == 'enter':
            plt.close(fig)
        update_display()

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.tight_layout()
    plt.show(block=True) # Ensure window blocks execution

    return [b for i, b in enumerate(blobs) if selected[i]]


class BlobDetectorGUI:
    """A class to handle the interactive blob detection GUI."""

    def __init__(self, img_crop: np.ndarray, use_preprocessing: bool, 
                 invert_threshold: bool, median_k: int, min_contour_area: int):
        self.img_crop = img_crop
        self.use_preprocessing = use_preprocessing
        self.invert_threshold = invert_threshold
        self.median_k = median_k
        self.min_contour_area = min_contour_area

        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.4)

        # Initial parameters
        self.clahe_clip = 2.0
        self.tophat_kernel = 15
        self.min_diam = 5
        self.max_diam = 50
        self.thresh_val = 128
        self.morph_iter = 2

        # Create sliders
        ax_clahe = plt.axes([0.15, 0.3, 0.65, 0.03])
        ax_tophat = plt.axes([0.15, 0.25, 0.65, 0.03])
        ax_min = plt.axes([0.15, 0.2, 0.65, 0.03])
        ax_max = plt.axes([0.15, 0.15, 0.65, 0.03])
        ax_thresh = plt.axes([0.15, 0.1, 0.65, 0.03])
        ax_morph = plt.axes([0.15, 0.05, 0.65, 0.03])

        self.s_clahe = Slider(ax_clahe, 'CLAHE Clip', 0.5, 10, valinit=self.clahe_clip, valstep=0.5)
        self.s_tophat = Slider(ax_tophat, 'Top-hat Kernel', 1, 100, valinit=self.tophat_kernel, valstep=1)
        self.s_min = Slider(ax_min, 'Min Diameter', 1, 150, valinit=self.min_diam, valstep=1)
        self.s_max = Slider(ax_max, 'Max Diameter', 10, 400, valinit=self.max_diam, valstep=1)
        self.s_thresh = Slider(ax_thresh, 'Threshold', 1, 250, valinit=self.thresh_val, valstep=1)
        self.s_morph = Slider(ax_morph, 'Morph Iter', 1, 10, valinit=self.morph_iter, valstep=1)

        ax_button = plt.axes([0.8, 0.01, 0.15, 0.04])
        self.button = Button(ax_button, 'Continue')

        # Connect events
        self.s_clahe.on_changed(self.update)
        self.s_tophat.on_changed(self.update)
        self.s_min.on_changed(self.update)
        self.s_max.on_changed(self.update)
        self.s_thresh.on_changed(self.update)
        self.s_morph.on_changed(self.update)
        self.button.on_clicked(self.finish)

        self.finished = False
        self.blobs: List[Dict[str, Any]] = []
        self.processed_img: Optional[np.ndarray] = None

        self.update(None) # Initial display

    def update(self, val: Any):
        """Callback to update processing and display when sliders change."""
        self.clahe_clip = self.s_clahe.val
        self.tophat_kernel = int(self.s_tophat.val)
        self.min_diam = self.s_min.val
        self.max_diam = self.s_max.val
        self.thresh_val = self.s_thresh.val
        self.morph_iter = int(self.s_morph.val)

        try:
            if self.use_preprocessing:
                # NOTE: This GUI is hard-coded to use 'preprocess_image_for_dark_spots'.
                # You could make the preprocessing function a parameter for more flexibility.
                self.processed_img = preprocess_image_for_dark_spots(
                    self.img_crop, 
                    clahe_clip=self.clahe_clip, 
                    tophat_kernel=self.tophat_kernel,
                    median_k=self.median_k, 
                    sigma=1.0,
                    scratch_removal=False
                )
            else:
                if self.img_crop.dtype != np.uint8:
                    self.processed_img = (self.img_crop / self.img_crop.max() * 255).astype(np.uint8)
                else:
                    self.processed_img = self.img_crop.copy()

            # Detect blobs
            img_uint8, _, self.blobs = detect_blobs_in_processed_image(
                self.processed_img,
                thresh_val=int(self.thresh_val),
                morph_iter=self.morph_iter,
                min_diam=self.min_diam,
                max_diam=self.max_diam,
                min_contour_area=self.min_contour_area,
                invert_threshold=self.invert_threshold
            )

            # Update display
            self.ax.clear()
            self.ax.imshow(img_uint8, cmap='gray')
            for b in self.blobs:
                if b["ellipse"] is not None:
                    e = Ellipse(b["center"], b["ellipse"][1][0], b["ellipse"][1][1],
                                angle=b["ellipse"][2], edgecolor='r', facecolor='none', lw=1)
                    self.ax.add_patch(e)
                else:
                    c = Circle(b["center"], radius=b["diam"] / 2, edgecolor='r', facecolor='none', lw=1)
                    self.ax.add_patch(c)

            self.ax.set_title(f'Detected {len(self.blobs)} spots')
            self.ax.axis('off')
            self.fig.canvas.draw_idle()

        except Exception as e:
            print(f"Error in GUI update: {e}")

    def finish(self, event: Any):
        """Callback for the 'Continue' button."""
        self.finished = True
        plt.close(self.fig)

    def show(self) -> Tuple[Optional[np.ndarray], List[Dict[str, Any]]]:
        """Show the GUI and block until 'Continue' is pressed."""
        plt.show(block=True)
        return self.processed_img, self.blobs


def detect_and_manual(
    img_crop: np.ndarray, 
    use_preprocessing: bool, 
    invert_threshold: bool, 
    median_k: int, 
    min_contour_area: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[Dict[str, Any]]]:
    """Main function for blob detection with interactive tuning and manual selection."""
    print("Opening interactive blob detection...")

    gui = BlobDetectorGUI(
        img_crop, 
        use_preprocessing=use_preprocessing, 
        invert_threshold=invert_threshold, 
        median_k=median_k, 
        min_contour_area=min_contour_area
    )
    processed_img, blobs = gui.show()

    if not blobs:
        print("No blobs detected with current parameters.")
        return processed_img, None, []

    print(f"Detected {len(blobs)} blobs. Opening manual selection...")
    if processed_img is None:
         # Fallback if processing failed
        processed_img = (img_crop / img_crop.max() * 255).astype(np.uint8)

    selected_blobs = manual_deselect(processed_img, blobs)

    print(f"Selected {len(selected_blobs)} blobs after manual filtering.")
    # The 'thresh' image is not used downstream, so returning None
    return processed_img, None, selected_blobs