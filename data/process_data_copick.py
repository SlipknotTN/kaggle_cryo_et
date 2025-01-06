"""
Save annotations for each experiment as 3D numpy arrays
"""
import argparse
import os
import shutil
from typing import Tuple

import copick
import numpy as np
import scipy.ndimage as ndimage
import zarr
from tqdm import tqdm

from shared.processing import remove_repeated_picks


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


def extract_coords(
    particle_metadata,
    labelmap,
    voxel_size=10,
    min_protein_size=0.4,
    remove_index=0,
):
    label = particle_metadata["label"]
    label_objs, num_objs = ndimage.label(labelmap == label)

    # Filter Candidates based on Object Size
    # Get the sizes of all objects
    object_sizes = np.bincount(label_objs.flat)

    # Filter the objects based on size
    min_object_size = (
        4
        / 3
        * np.pi
        * ((particle_metadata["radius"] / voxel_size) ** 2)
        * min_protein_size
    )
    valid_objects = np.where(object_sizes > min_object_size)[0]

    # Estimate Coordiantes from CoM for LabelMaps
    deepFinderCoords = []
    for object_num in tqdm(valid_objects):
        com = ndimage.center_of_mass(label_objs == object_num)
        swapped_com = (com[2], com[1], com[0])
        deepFinderCoords.append(swapped_com)
    deepFinderCoords = np.array(deepFinderCoords)

    # For some reason, consistently extracting center coordinate
    # Remove the row with the closest index
    deepFinderCoords = np.delete(deepFinderCoords, remove_index, axis=0)

    # Estimate Distance Threshold Based on 1/2 of Particle Diameter
    threshold = np.ceil(particle_metadata["radius"] / (voxel_size * 3))

    try:
        # Remove Double Counted Coordinates
        deepFinderCoords = remove_repeated_picks(deepFinderCoords, threshold)

        # Convert from Voxel to Physical Units
        deepFinderCoords *= voxel_size

    except Exception as e:
        print(f"Error processing label {label} in tomo: {e}")
        deepFinderCoords = np.array([]).reshape(0, 6)

    return deepFinderCoords


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

        for particle_name, particle_metadata in tqdm(
            particles.items(), desc="particle"
        ):
            picks = run.get_picks(
                particle_name, user_id="curation", session_id="0"
            )
            assert len(picks) == 1

            print(f"Experiment {run.name}, particle {particle_name}: found {len(picks[0].points)} points")
            # Draw sphere manually setting to label every point at distance <= radius from the center
            # Do the comparison in a cube around the center to speed up the comparison
            for point in picks[0].points:
                # Get location coordinates and radius converting from Angstrom to voxel coordinates
                point_x, point_y, point_z = (
                    round(point.location.x / args.voxel_spacing),
                    round(point.location.y / args.voxel_spacing),
                    round(point.location.z / args.voxel_spacing),
                )
                radius = round(
                    particles[particle_name]["radius"] / args.voxel_spacing
                )
                # Crop annotations_zyx around the cube around the sphere
                min_cube_point_z = max(0, point_z - radius)
                max_cube_point_z = min(
                    annotations_zyx.shape[0] - 1, point_z + radius
                )
                min_cube_point_y = max(0, point_y - radius)
                max_cube_point_y = min(
                    annotations_zyx.shape[1] - 1, point_y + radius
                )
                min_cube_point_x = max(0, point_x - radius)
                max_cube_point_x = min(
                    annotations_zyx.shape[2] - 1, point_x + radius
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
                    radius=radius,
                    label=particles[particle_name]["label"],
                )

        # TODO: Trying to extract again the centroid from the sphere. It doesn't work,
        # until this is not fixed we can't extract the inference results correctly

        for particle_name, particle_metadata in tqdm(
            particles.items(), desc="particle"
        ):
            re_extracted_coords = extract_coords(
                particle_metadata, annotations_zyx
            )
            print(f"Experiment {run.name}, particle {particle_name}: re-extracted coords {len(re_extracted_coords)}")
            for coord in re_extracted_coords:
                print(f"Particle name {particle_name}: {coord}")

        output_npy_filepath = os.path.join(args.output_dir, f"{run.name}.npy")
        os.makedirs(os.path.dirname(output_npy_filepath), exist_ok=True)
        np.save(output_npy_filepath, annotations_zyx)
        print(f"Saved {output_npy_filepath}")


if __name__ == "__main__":
    main()
