"""
Save annotations for each experiment as 3D numpy arrays
"""
import argparse
import os
import shutil
from typing import List, Tuple

import copick
import numpy as np
import scipy.ndimage as ndi
import torch
import zarr
from skimage.measure import regionprops
from skimage.segmentation import watershed
from tqdm import tqdm

from shared.processing import (
    calc_distances_matrix,
    create_hessian_particle_mask,
    distance_transform,
    erode_dilate_mask,
    local_maxima,
    remove_repeated_picks,
)


def set_sphere_to_label(
    cube_zyx_inout: np.ndarray,
    center_zyx: Tuple[int, int, int],
    radius: int,
    label: int,
):
    dim_zyx = cube_zyx_inout.shape
    for z in range(dim_zyx[0]):
        for y in range(dim_zyx[1]):
            for x in range(dim_zyx[2]):
                # Square normalized distance (1 means "radius" distance)
                square_norm_distance = (
                    ((z - center_zyx[0]) / radius) ** 2
                    + ((y - center_zyx[1]) / radius) ** 2
                    + ((x - center_zyx[2]) / radius) ** 2
                )
                if square_norm_distance <= 1.0:
                    cube_zyx_inout[z][y][x] = label


def extract_coords_from_volume(
    particle_metadata,
    labelmap,
    radius,
    voxel_size=10,
    min_protein_size=0.4,
    remove_index=0,
):
    """
    Extract particle coordinates from labeled volume (each value corresponds to a particle label)

    The logic is based on detecting the center of mass of each structured element in the volume
    """
    label = particle_metadata["label"]
    label_objs, num_objs = ndi.label(labelmap == label)

    # Filter Candidates based on Object Size
    # Get the sizes of all objects
    object_sizes = np.bincount(label_objs.flat)

    # Filter the objects based on size
    min_object_size = (
        4 / 3 * np.pi * ((radius / voxel_size) ** 2) * min_protein_size
    )
    valid_objects = np.where(object_sizes > min_object_size)[0]

    # Estimate Coordiantes from CoM for LabelMaps
    deepFinderCoords = []
    for object_num in tqdm(valid_objects):
        com = ndi.center_of_mass(label_objs == object_num)
        swapped_com = (com[2], com[1], com[0])
        deepFinderCoords.append(swapped_com)
    deepFinderCoords = np.array(deepFinderCoords)

    # For some reason, consistently extracting center coordinate
    # Remove the row with the closest index
    deepFinderCoords = np.delete(deepFinderCoords, remove_index, axis=0)

    # Estimate Distance Threshold Based on 1/2 of Particle Diameter
    threshold = np.ceil(radius / (voxel_size * 3))

    try:
        # Remove Double Counted Coordinates
        deepFinderCoords = remove_repeated_picks(deepFinderCoords, threshold)

        # Convert from Voxel to Physical Units
        deepFinderCoords *= voxel_size

    except Exception as e:
        print(f"Error processing label {label} in tomo: {e}")
        deepFinderCoords = np.array([]).reshape(0, 6)

    return deepFinderCoords


def extract_coords_from_volume_v2(
    volume: torch.Tensor, scaled_radius, particle_metadata, device="cuda"
) -> List[np.ndarray]:
    # TODO: Not working with spheres as annotations, but it works with tomograms_tensors + binary_mask from hessian
    # in the baseline script. Here it always returns an empty list of centroids

    volume = volume.to(device)

    label = particle_metadata["label"]

    # Filter the volume on the specific particle (not strictly necessary, we could evaluate all particles together)
    volume_particle = torch.where(volume == label, 1.0, 0.0)

    if torch.sum(volume_particle) == 0:
        return []

    # Erode and dilate the segmentation
    dilated_mask = erode_dilate_mask(
        volume_particle, scaled_radius, device=device
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
    watershed_labels = watershed(-distance_np, markers, mask=dilated_mask_np)

    # Extract region properties and scale coordinates back to original space
    centroids = []
    for region in regionprops(watershed_labels):
        # Scale the centroid coordinates back to original space
        centroid = np.array(region.centroid)
        centroids.append(centroid)
    return centroids


def do_parsing():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Process data with Copick and convert to numpy array",
    )
    parser.add_argument(
        "--copick_config_path",
        required=True,
        type=str,
        help="Copick configuration file, there should be one for train and one for test",
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
        "--annotation_type",
        required=True,
        type=str,
        choices=["point", "sphere"],
        help="Particle annotation shape: exact ground truth point or sphere around the point with the particle radius",
    )
    parser.add_argument(
        "--fixed_sphere_radius",
        required=False,
        type=float,
        help="Set sphere radius to a fixed value instead of particle radius. Radius value in the original voxel space.",
    )
    parser.add_argument(
        "--test_extraction_from_spheres",
        action="store_true",
        help="Test the centroids extraction from spheres as annotations",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="Output dir to save the input and the labels as numpy arrays",
    )
    args = parser.parse_args()
    return args


def main():
    args = do_parsing()
    print(args)
    root_copick = copick.from_file(args.copick_config_path)
    particles = dict()
    for obj in tqdm(root_copick.pickable_objects, desc="object"):
        if not obj.is_particle:
            continue
        print(f"Name: {obj.name}, radius: {obj.radius}")
        particles[obj.name] = {"radius": obj.radius, "label": obj.label}

    # Fix overlay json names
    for root, dirs, files in os.walk(root_copick.config.overlay_root):
        # Copy and rename each file
        for file in files:
            if file.startswith("curation_0_"):
                continue
            else:
                new_filename = f"curation_0_{file}"
                # Define full paths for the source and destination files
                source_file = os.path.join(root, file)
                destination_file = os.path.join(root, new_filename)
                shutil.move(source_file, destination_file)
                print(f"Renamed {source_file} to {destination_file}")

    for run in tqdm(root_copick.runs, desc="experiment_run"):
        # Read tomograms as numpy array
        tomogram_wrapper = run.get_voxel_spacing(
            args.voxel_spacing
        ).get_tomogram(args.tomo_type)
        z = zarr.open(store=tomogram_wrapper.zarr(), path="/", mode="r")
        high_res_tomogram_zyx = z["0"][:]
        print(
            f"Run {run.name}, high res tomogram shape {high_res_tomogram_zyx.shape}"
        )

        annotations_zyx = np.zeros(
            shape=high_res_tomogram_zyx.shape, dtype=np.uint8
        )

        points_per_particle = dict()

        for particle_name, particle_metadata in tqdm(
            particles.items(), desc="particle"
        ):
            picks = run.get_picks(
                particle_name, user_id="curation", session_id="0"
            )
            assert len(picks) == 1

            print(
                f"Experiment {run.name}, particle {particle_name}: found {len(picks[0].points)} points"
            )

            points_per_particle[particle_name] = len(picks[0].points)

            # Radius in voxel coordinates
            radius = (
                particles[particle_name]["radius"]
                if args.fixed_sphere_radius
                else args.fixed_sphere_radius
            )
            radius_voxel = round(radius / args.voxel_spacing)

            # Draw sphere manually setting to label every point at distance <= radius from the center
            # Do the comparison in a cube around the center to speed up the comparison
            for point in picks[0].points:
                # Get location coordinates and radius converting from Angstrom to voxel coordinates
                point_x, point_y, point_z = (
                    round(point.location.x / args.voxel_spacing),
                    round(point.location.y / args.voxel_spacing),
                    round(point.location.z / args.voxel_spacing),
                )
                if args.annotation_type == "sphere":
                    # Crop annotations_zyx around the cube around the sphere
                    min_cube_point_z = max(0, point_z - radius_voxel)
                    max_cube_point_z = min(
                        annotations_zyx.shape[0] - 1, point_z + radius_voxel
                    )
                    min_cube_point_y = max(0, point_y - radius_voxel)
                    max_cube_point_y = min(
                        annotations_zyx.shape[1] - 1, point_y + radius_voxel
                    )
                    min_cube_point_x = max(0, point_x - radius_voxel)
                    max_cube_point_x = min(
                        annotations_zyx.shape[2] - 1, point_x + radius_voxel
                    )
                    cube = annotations_zyx[
                        min_cube_point_z:max_cube_point_z,
                        min_cube_point_y:max_cube_point_y,
                        min_cube_point_x:max_cube_point_x,
                    ]
                    cube_center_zyx = (
                        point_z - min_cube_point_z,
                        point_y - min_cube_point_y,
                        point_x - min_cube_point_x,
                    )
                    # First one preferred
                    set_sphere_to_label(
                        cube_zyx_inout=cube,
                        center_zyx=cube_center_zyx,
                        radius=radius_voxel,
                        label=particles[particle_name]["label"],
                    )

                elif args.annotation_type == "point":
                    annotations_zyx[point_z][point_y][point_x] = particles[
                        particle_name
                    ]["label"]

                else:
                    raise Exception(
                        f"Annotation type {args.annotation_type} not supported"
                    )

            # Calculate distances between all points of the same particle
            distances_matrix = calc_distances_matrix(picks[0].points)
            # Identify spheres around particles that overlaps
            # (use triangular matrix to avoid returning the opposite indexes couple)
            distances_matrix_triangular = np.triu(distances_matrix, 0)
            closest_points_indexes = np.argwhere(
                (distances_matrix_triangular > 0.0)
                & (distances_matrix_triangular <= radius * 2)
            )
            if len(closest_points_indexes) > 0:
                print(
                    f"WARNING: Experiment {run.name}, particle {particle_name} "
                    f"has these points closest than (radius x 2): {closest_points_indexes}"
                )

        # Trying to extract again the centroid from the sphere.
        # It doesn't work when the points are too close (less than radius distance)
        # and they recognized as a single object
        if (
            args.annotation_type == "sphere"
            and args.test_extraction_from_spheres
        ):
            for particle_name, particle_metadata in tqdm(
                particles.items(), desc="particle"
            ):
                re_extracted_coords = extract_coords_from_volume(
                    particle_metadata,
                    annotations_zyx,
                    radius=particle_metadata["radius"]
                    if args.fixed_sphere_radius is False
                    else args.fixed_sphere_radius,
                )
                print(
                    f"Experiment {run.name}, particle {particle_name}: "
                    f"re-extracted points from sphere "
                    f"{len(re_extracted_coords)} Vs {points_per_particle[particle_name]} originally"
                )

                re_extracted_coords_v2 = extract_coords_from_volume_v2(
                    volume=torch.tensor(annotations_zyx),
                    scaled_radius=round(
                        particle_metadata["radius"] / args.voxel_spacing
                    )
                    if args.fixed_sphere_radius is False
                    else round(args.fixed_sphere_radius / args.voxel_spacing),
                    particle_metadata=particle_metadata,
                )
                print(
                    f"Experiment {run.name}, particle {particle_name}: "
                    f"re-extracted points from sphere V2 "
                    f"{len(re_extracted_coords_v2)} Vs {points_per_particle[particle_name]} originally"
                )

        output_npy_filepath = os.path.join(args.output_dir, f"{run.name}.npy")
        os.makedirs(os.path.dirname(output_npy_filepath), exist_ok=True)
        np.save(output_npy_filepath, annotations_zyx)
        print(f"Saved {output_npy_filepath}")


if __name__ == "__main__":
    main()
