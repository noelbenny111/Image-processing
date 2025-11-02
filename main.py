"""
main.py

Main script for running laser spot analysis on CZI files.
Reads configuration, determines the processing mode (single 5x or batch),
and executes the analysis pipeline.
"""

import os
import glob
from typing import Dict, Any

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle

# Import our custom utility functions
import analysis_utils as utils

# ------------------ User Parameters ------------------
# This is now your main configuration area.

# Set the processing mode
PROCESS_5X_IMAGE = False

# Parameters for 5x single image mode
if PROCESS_5X_IMAGE:
    CZI_PATH = r"C:\Users\n.benny\Desktop\Data\Microsocope images\130825\140825_CG_Z_0.95\5x_czi files\29-36_right-left_5X_circle.czi"
    OUTPUT_FOLDER = os.path.join(os.path.dirname(CZI_PATH), "processed_results_5x_29-36-4")
    DEG_START, DEG_END = 36, 29
    COLUMN_WIDTH = 150
# Parameters for batch processing mode
else:
    FOLDER_PATH = r"C:\Users\n.benny\Desktop\Data\Microsocope images\180925_focus\Focus-5,65-09,0925"
    OUTPUT_FOLDER = os.path.join(FOLDER_PATH, "processed_results_3")
    X_START, WIDTH = 400, 800

# Common detection and analysis parameters
DETECTION_PARAMS = {
    "median_kernel": 3,
    "reference_diameter_um": None,  # Set this to your known reference diameter in µm
    "USE_PREPROCESSING": True,
    "INVERT_THRESHOLD": True,       # True for bright spots on dark (inverted) background
    "MIN_CONTOUR_AREA": 20
}

# ------------------ Main Processing Functions ------------------

def process_single_5x_image(
    czi_path: str, 
    output_folder: str, 
    deg_start: int, 
    deg_end: int, 
    column_width: int, 
    detection_params: Dict[str, Any]
) -> int:
    """Process a single 5x image by splitting it into degree columns."""
    print(f"Processing single 5x image: {czi_path}")
    os.makedirs(output_folder, exist_ok=True)

    img = utils.load_czi_image(czi_path)
    pixel_size_um = utils.get_pixel_size_um_from_czi(czi_path)
    if pixel_size_um is None:
        print("WARNING: Pixel size not found. Using fallback 0.34 µm.")
        pixel_size_um = 0.34
    print(f"Pixel size: {pixel_size_um:.4f} µm")

    # Interactive column width selection
    current_column_width = column_width
    confirmed = False
    while not confirmed:
        try:
            user_input = input(f"Current column width = {current_column_width}px. Press Enter to keep, or enter new value: ").strip()
            if user_input:
                current_column_width = int(user_input)
        except ValueError:
            print("Invalid input, keeping current width.")
        
        degree_regions, _ = utils.extract_degree_columns(img, deg_start, deg_end, current_column_width)
        print(f"Extracted {len(degree_regions)} degree regions")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(img, cmap="gray")
        for deg, region in degree_regions.items():
            rect = plt.Rectangle((region['x_min'], 0), region['x_max'] - region['x_min'], img.shape[0],
                                 linewidth=2, edgecolor="red", facecolor="none")
            ax.add_patch(rect)
            ax.text(region['x_center'], 10, str(deg), color="yellow", ha="center", va="top",
                    fontsize=8, weight="bold", bbox=dict(facecolor="black", alpha=0.5, pad=1))
        ax.set_title(f"Original image with cropped regions (width={current_column_width}px)")
        plt.show(block=True)

        ans = input("Are the regions correct? (y/n): ").strip().lower()
        if ans == "y":
            confirmed = True
        else:
            print("❌ Regions not confirmed. Please enter a new column width.")

    # Process each region
    results = []
    all_blob_data = []

    for deg, region_info in degree_regions.items():
        print(f"\n=== Processing degree {deg} ===")
        strip = region_info['strip']

        processed_img, _, selected_blobs = utils.detect_and_manual(
            strip,
            use_preprocessing=detection_params["USE_PREPROCESSING"],
            invert_threshold=detection_params["INVERT_THRESHOLD"],
            median_k=detection_params["median_kernel"],
            min_contour_area=detection_params["MIN_CONTOUR_AREA"]
        )

        if selected_blobs:
            diameters_px = [b["diam"] for b in selected_blobs]
            diameters_um = [d * pixel_size_um for d in diameters_px]
            error_metrics = utils.calculate_measurement_errors(diameters_um, detection_params["reference_diameter_um"])

            results.append({
                "degree": deg,
                "x_center_px": region_info['x_center'],
                "avg_diameter_px": np.mean(diameters_px),
                "avg_diameter_um": error_metrics["mean"],
                "n_blobs": len(diameters_um),
                "std_diameter_um": error_metrics["std"],
                "cv_percent": error_metrics["cv_percent"],
                "ci_lower": error_metrics["confidence_interval"][0],
                "ci_upper": error_metrics["confidence_interval"][1]
            })

            # Save annotated image
            if processed_img is not None:
                annotated = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
                for b in selected_blobs:
                    if b["ellipse"] is not None:
                        cv2.ellipse(annotated, b["ellipse"], (0, 255, 0), 2)
                    else:
                        center = (int(b["center"][0]), int(b["center"][1]))
                        radius = int(b["diam"] / 2)
                        cv2.circle(annotated, center, radius, (0, 255, 0), 2)
                out_img_path = os.path.join(output_folder, f"degree_{deg}_annotated.png")
                cv2.imwrite(out_img_path, annotated)

            for j, b in enumerate(selected_blobs):
                all_blob_data.append({
                    "degree": deg, "blob_id": j, "x_px": b["center"][0], "y_px": b["center"][1],
                    "diameter_px": b["diam"], "diameter_um": diameters_um[j]
                })

            print(f"Degree {deg}: {len(diameters_um)} spots, avg diameter: {error_metrics['mean']:.2f} µm")
    
    # Save summary files
    if results:
        df_results = pd.DataFrame(results).sort_values("degree", ascending=False)
        df_results.to_csv(os.path.join(output_folder, "degree_measurements_summary.csv"), index=False)
        
        df_blobs = pd.DataFrame(all_blob_data)
        df_blobs.to_csv(os.path.join(output_folder, "all_blobs_detailed.csv"), index=False)
        
        print("\nFinal Summary:")
        print(df_results.to_string(index=False))

    return len(results)


def process_batch_images(
    folder_path: str, 
    output_folder: str, 
    x_start: int, 
    width: int, 
    detection_params: Dict[str, Any]
) -> int:
    """Process multiple CZI files in batch mode."""
    os.makedirs(output_folder, exist_ok=True)
    czi_files = glob.glob(os.path.join(folder_path, "*.czi"))
    print(f"Found {len(czi_files)} CZI files for batch processing...")

    summary_data = []

    for i, czi_path in enumerate(czi_files):
        print(f"\n=== Processing file {i + 1}/{len(czi_files)}: {os.path.basename(czi_path)} ===")
        
        degree = utils.extract_degree_from_filename(czi_path)
        if degree is None:
            continue

        try:
            img = utils.load_czi_image(czi_path)
            img_crop = utils.crop_image(img, x_start, width)

            pixel_size_um = utils.get_pixel_size_um_from_czi(czi_path)
            if pixel_size_um is None:
                print(f"Warning: Skipping {czi_path} - could not extract pixel size.")
                continue
            print(f"Pixel size: {pixel_size_um:.4f} µm")

            processed_img, _, blobs_final = utils.detect_and_manual(
                img_crop,
                use_preprocessing=detection_params["USE_PREPROCESSING"],
                invert_threshold=detection_params["INVERT_THRESHOLD"],
                median_k=detection_params["median_kernel"],
                min_contour_area=detection_params["MIN_CONTOUR_AREA"]
            )

            if blobs_final:
                diameters_px = [b["diam"] for b in blobs_final]
                diameters_um = [d * pixel_size_um for d in diameters_px]
                error_metrics = utils.calculate_measurement_errors(diameters_um, detection_params["reference_diameter_um"])

                summary_data.append({
                    "degree": degree,
                    "avg_diameter_px": np.mean(diameters_px),
                    "avg_diameter_um": error_metrics["mean"],
                    "n_blobs": len(diameters_um),
                    "std_diameter_um": error_metrics["std"],
                    "cv_percent": error_metrics["cv_percent"],
                    "ci_lower": error_metrics["confidence_interval"][0],
                    "ci_upper": error_metrics["confidence_interval"][1]
                })

                print(f"Degree {degree}: {len(diameters_um)} spots, avg diameter: {error_metrics['mean']:.2f} µm")

                # Save annotated image
                if processed_img is not None:
                    fig, ax = plt.subplots()
                    ax.imshow(processed_img, cmap='gray')
                    for b in blobs_final:
                        if b["ellipse"] is not None:
                            e = Ellipse(b["center"], b["ellipse"][1][0], b["ellipse"][1][1],
                                        angle=b["ellipse"][2], edgecolor='lime', facecolor='none', lw=1)
                            ax.add_patch(e)
                        else:
                            c = Circle(b["center"], radius=b["diam"] / 2, edgecolor='lime', facecolor='none', lw=1)
                            ax.add_patch(c)
                    ax.axis('off')
                    out_img_path = os.path.join(output_folder, os.path.basename(czi_path).replace(".czi", "_annotated.png"))
                    fig.savefig(out_img_path, dpi=200, bbox_inches='tight')
                    plt.close(fig)

                # Save individual blob data
                blob_data = [{
                    "blob_id": j, "x_px": b["center"][0], "y_px": b["center"][1],
                    "diameter_px": b["diam"], "diameter_um": diameters_um[j]
                } for j, b in enumerate(blobs_final)]
                
                df = pd.DataFrame(blob_data)
                out_csv_path = os.path.join(output_folder, os.path.basename(czi_path).replace(".czi", "_blobs.csv"))
                df.to_csv(out_csv_path, index=False)

            else:
                print(f"No blobs found for {os.path.basename(czi_path)}")

        except Exception as e:
            print(f"Error processing {czi_path}: {e}")
            continue

    # Save batch summary
    if summary_data:
        summary_df = pd.DataFrame(summary_data).sort_values("degree", ascending=False)
        summary_csv = os.path.join(output_folder, "batch_measurements_summary.csv")
        summary_df.to_csv(summary_csv, index=False)

        print(f"\n=== BATCH PROCESSING COMPLETE ===")
        print(f"Summary saved: {summary_csv}")
        print("\nSummary Table:")
        print(summary_df.to_string(index=False))
        
    return len(summary_data)

# ------------------ Script Execution ------------------

def main():
    """Main execution function to run the correct processing pipeline."""
    print("=== Enhanced Laser Spot Analysis ===")
    
    if PROCESS_5X_IMAGE:
        success_count = process_single_5x_image(
            CZI_PATH, OUTPUT_FOLDER, DEG_START, DEG_END, COLUMN_WIDTH, DETECTION_PARAMS
        )
        print(f"\nProcessing complete! Analyzed {success_count} degrees.")
    else:
        success_count = process_batch_images(
            FOLDER_PATH, OUTPUT_FOLDER, X_START, WIDTH, DETECTION_PARAMS
        )
        print(f"\nBatch processing complete! Processed {success_count} files.")

    print(f"All results saved to: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()