import argparse
import os

import copick
import cv2
import numpy as np
from tqdm import tqdm
import zarr

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
        help="Voxel spacing used to produce the data"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="Output dir to save the input and the labels as numpy arrays"
    )
    args = parser.parse_args()
    return args


def main():
    args = do_parsing()
    print(args)
    root = copick.from_file(args.copick_config_path)
    particles = dict()
    for obj in tqdm(root.pickable_objects, desc="object"):
        if not obj.is_particle:
            continue
        print(f"Name: {obj.name}, radius: {obj.radius}")
        particles[obj.name] = {"radius": obj.radius}
    # TODO: Split visualization util code or add under debug mode
    for run in root.runs:
        tomogram_wrapper = run.get_voxel_spacing(args.voxel_spacing).get_tomogram(args.tomo_type)
        z = zarr.open(store=tomogram_wrapper.zarr(), path="/", mode="r")
        for ix in ['0', '1', '2']:
            high_res_zyx = z[ix][:]
            print(f"Run {run.name}, high res shape {high_res_zyx.shape}")
            slice_id = 0
            max_high_res = np.max(high_res_zyx[slice_id])
            min_high_res = np.min(high_res_zyx[slice_id])
            high_res_norm = (high_res_zyx[slice_id] - min_high_res) / (max_high_res - min_high_res)
            cv2.imshow(ix, high_res_norm)
        cv2.waitKey(0)

        # TODO: Load annotations for each particle from files like ./dataset/train/overlay/ExperimentRuns/TS_5_4/Picks/apo-ferritin.json
        # TODO: Extend point with radius considering the voxel spacing

if __name__ == "__main__":
    main()
