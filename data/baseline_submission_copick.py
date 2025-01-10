"""
Code copied from the official notebook https://www.kaggle.com/code/kharrington/blobdetector
Runtime errors about types are expected

Create a submission CSV starting from copick format
"""

import argparse
import csv
import os
import time

import copick
import numpy as np
import scipy.ndimage as ndi
import torch
from copick.impl.filesystem import CopickRootFSSpec
from skimage.measure import regionprops
from skimage.segmentation import watershed
from tqdm import tqdm

from shared.processing import (
    create_hessian_particle_mask,
    distance_transform,
    erode_dilate_mask,
    get_tomogram_data,
    local_maxima,
)


def process_all_runs(
    root: CopickRootFSSpec,
    session_id: str,
    user_id: str,
    voxel_spacing: int,
    tomo_type: str,
    resolution_threshold: int,
    device: str,
    output_csv_path: str,
):
    """Process all runs and save results to CSV."""
    results = []
    pick_id = 0

    for run in tqdm(root.runs):
        start_time = time.time()
        print(f"\nProcessing run: {run.meta.name}")

        # Process each particle type separately since they might need different resolutions
        for obj in root.pickable_objects:
            if not obj.is_particle:
                continue

            radius = obj.radius
            print(f"Processing {obj.name} with radius {radius}")

            # Get appropriate resolution data
            (
                tomogram_tensor,
                effective_voxel_spacing,
                scale_factor,
            ) = get_tomogram_data(
                run,
                voxel_spacing,
                radius,
                tomo_type=tomo_type,
                device=device,
                resolution_threshold=resolution_threshold,
            )

            print(
                f"Using scale factor {scale_factor} (effective voxel spacing: {effective_voxel_spacing})"
            )

            # Create segmentation at appropriate scale
            segmentation = create_hessian_particle_mask(
                tomogram_tensor, sigma=3, device=device
            )

            if torch.sum(segmentation) == 0:
                print(f"No particles detected in segmentation for {obj.name}")
                continue

            # Adjust radius for effective voxel spacing
            scaled_radius = radius / effective_voxel_spacing

            scaled_radius = round(scaled_radius)

            # Erode and dilate the segmentation
            dilated_mask = erode_dilate_mask(
                segmentation, scaled_radius, device=device
            )

            # Distance transform and local maxima detection
            distance = distance_transform(dilated_mask, device=device)
            local_max = local_maxima(distance, scaled_radius)

            # Convert tensors to numpy for watershed
            local_max_np = local_max.cpu().numpy()
            distance_np = distance.cpu().numpy()
            dilated_mask_np = dilated_mask.cpu().numpy()

            # Watershed segmentation
            markers, _ = ndi.label(local_max_np)
            watershed_labels = watershed(
                -distance_np, markers, mask=dilated_mask_np
            )

            # Extract region properties and scale coordinates back to original space
            centroids = []
            for region in regionprops(watershed_labels):
                # Scale the centroid coordinates back to original space
                centroid = np.array(region.centroid) * scale_factor
                centroids.append(centroid)  # ZYX order

            # Save centroids as picks and add to results
            if centroids:
                pick_set = run.get_picks(obj.name)
                if pick_set:
                    pick_set = pick_set[0]
                else:
                    pick_set = run.new_picks(obj.name, session_id, user_id)

                for centroid in centroids:
                    # Convert from ZYX to XYZ order and apply voxel spacing
                    x = centroid[2] * voxel_spacing  # Z -> X
                    y = centroid[1] * voxel_spacing  # Y -> Y
                    z = centroid[0] * voxel_spacing  # X -> Z

                    # Add to results list
                    row = [pick_id, run.meta.name, obj.name, x, y, z]
                    results.append(row)
                    pick_id += 1

                # Store pick set
                pick_set.points = [
                    {
                        "x": c[2] * voxel_spacing,
                        "y": c[1] * voxel_spacing,
                        "z": c[0] * voxel_spacing,
                    }
                    for c in centroids
                ]
                pick_set.store()
                print(f"Saved {len(centroids)} centroids for {obj.name}")
            else:
                print(f"No valid centroids found for {obj.name}")

        # Print timing for this run
        end_time = time.time()
        print(
            f"Run {run.meta.name} completed in {end_time - start_time:.2f} seconds"
        )

    print(f"\nTotal picks found: {len(results)}")

    # Write results to CSV
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "experiment", "particle_type", "x", "y", "z"])
        writer.writerows(results)

    print(f"Results saved to {output_csv_path}")
    return results


def do_parsing():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Produce the baseline submission CSV processing the data with Copick",
    )
    parser.add_argument(
        "--copick_config_path",
        required=False,
        type=str,
        default="./dataset/copick/copick.config",
        help="Copick configuration file",
    )
    parser.add_argument(
        "--tomo_type",
        required=False,
        default="denoised",
        type=str,
        help="Tomograph type",
    )
    parser.add_argument(
        "--voxel_spacing",
        required=False,
        default=10,
        help="Voxel spacing used to produce the data",
    )
    parser.add_argument(
        "--resolution_threshold",
        required=False,
        default=16,
        type=int,
        help="Resolution threshold to use high or medium resolution",
    )
    parser.add_argument(
        "--device",
        required=False,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device to use",
    )
    parser.add_argument(
        "--output_csv_path",
        required=True,
        type=str,
        help="Output CSV path with predictions",
    )
    args = parser.parse_args()
    return args


def main():
    args = do_parsing()
    print(args)
    root = copick.from_file(args.copick_config_path)
    results = process_all_runs(
        root=root,
        session_id="0",
        user_id="blobDetector",
        voxel_spacing=args.voxel_spacing,
        tomo_type=args.tomo_type,
        resolution_threshold=args.resolution_threshold,
        device=args.device,
        output_csv_path=args.output_csv_path,
    )


if __name__ == "__main__":
    main()
