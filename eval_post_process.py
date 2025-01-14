"""
Script to evaluate the 3D annotations with the official metrics to measure the postprocessing
accuracy starting from the generated ground truth (e.g. sphere around points).

It assumes the process_data_copick.py was run at least one time, the overlay json files should be renamed
as curation_0_<particle_name>.json.
"""
import argparse
import os

import copick
import numpy as np
import pandas as pd
from tqdm import tqdm

from data.process_data_copick import extract_coords_from_volume
from metrics.metrics import score


def do_parsing():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Process ground truth data and measure post processing accuracy with the competition metric",
    )
    parser.add_argument(
        "--copick_config_path",
        required=True,
        type=str,
        help="Copick configuration file, there should be one for train and one for test",
    )
    parser.add_argument(
        "--input_ann_dir",
        required=True,
        type=str,
        help="Input dir with the ground truth 3D zyx numpy arrays for which we want to extract the point predictions",
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
    return parser.parse_args()


def main():
    args = do_parsing()
    print(args)

    assert (
        args.annotation_type == "sphere"
    ), f"Only annotation_type sphere is supported, points has no processing to be evaluated"

    # DataFrame with perfect ground truth from copick
    gt_df = pd.DataFrame(columns=["experiment", "particle_type", "x", "y", "z"])
    gt_df.index.name = "id"

    # DataFrame with the extracted results from the processed ground truth (e.g. points converted to spheres)
    post_process_df = pd.DataFrame(columns=gt_df.columns)
    post_process_df.index.name = "id"

    root_copick = copick.from_file(args.copick_config_path)
    particles = dict()
    for obj in tqdm(root_copick.pickable_objects, desc="object"):
        if not obj.is_particle:
            continue
        print(f"Name: {obj.name}, radius: {obj.radius}")
        particles[obj.name] = {"radius": obj.radius, "label": obj.label}

    for run in tqdm(root_copick.runs, desc="experiment_run"):
        for particle_name, particle_metadata in tqdm(
            particles.items(), desc="particle"
        ):
            picks = run.get_picks(
                particle_name, user_id="curation", session_id="0"
            )
            assert len(picks) == 1

            for point in picks[0].points:
                gt_point_row = pd.DataFrame.from_dict(
                    {
                        "experiment": [run.name],
                        "particle_type": [particle_name],
                        "x": [point.location.x],
                        "y": [point.location.y],
                        "z": [point.location.z],
                    }
                )
                if len(gt_df) == 0:
                    gt_df = gt_point_row
                else:
                    gt_df = pd.concat([gt_df, gt_point_row])

        processed_gt_zyx = np.load(
            os.path.join(args.input_ann_dir, f"{run.name}.npy")
        )
        for particle_name, particle_metadata in tqdm(
            particles.items(), desc="particle"
        ):
            extracted_points = extract_coords_from_volume(
                particle_metadata,
                processed_gt_zyx,
                radius=particle_metadata["radius"],
            )
            for extracted_point_zyx in extracted_points:
                x = extracted_point_zyx[2]
                y = extracted_point_zyx[1]
                z = extracted_point_zyx[0]
                post_process_point_row = pd.DataFrame.from_dict(
                    {
                        "experiment": [run.name],
                        "particle_type": [particle_name],
                        "x": [x],
                        "y": [y],
                        "z": [z],
                    }
                )
                if len(post_process_df) == 0:
                    post_process_df = post_process_point_row
                else:
                    post_process_df = pd.concat(
                        [post_process_df, post_process_point_row]
                    )

    # Save CSV for debug
    gt_df.to_csv(
        os.path.join(os.path.dirname(args.copick_config_path), "gt_points.csv")
    )
    post_process_df.to_csv(
        os.path.join(
            os.path.dirname(args.copick_config_path), "post_process_points.csv"
        )
    )
    print(f"CSV files saved to {os.path.dirname(args.copick_config_path)}")

    f4_score = score(
        solution=gt_df,
        submission=post_process_df,
        distance_multiplier=1.0,
        beta=4,
    )
    print(f"F4 score: {f4_score}")


if __name__ == "__main__":
    main()
