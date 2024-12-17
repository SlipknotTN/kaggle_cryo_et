import argparse
import os

import copick
import cv2
import numpy as np
import zarr
from tqdm import tqdm


def do_parsing():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Visualize tomograms and annotations as 2D slices",
    )
    parser.add_argument(
        "--copick_config_path",
        required=True,
        type=str,
        help="Copick configuration file, there should be one for train and one for test",
    )
    parser.add_argument(
        "--voxel_spacing",
        required=False,
        default=10,
        help="Voxel spacing used to produce the data",
    )
    parser.add_argument(
        "--tomo_type",
        required=False,
        default="denoised",
        type=str,
        help="Tomograph type",
    )
    parser.add_argument(
        "--annotations_dir",
        required=True,
        type=str,
        help="Annotations directory including one numpy file for each experiment run",
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
        particles[obj.name] = {
            "radius": obj.radius,
            "label": obj.label,
            "color": obj.color,
        }

    for run in tqdm(root_copick.runs, desc="experiment_run"):
        # Read tomograms as numpy array
        tomogram_wrapper = run.get_voxel_spacing(
            args.voxel_spacing
        ).get_tomogram(args.tomo_type)
        z = zarr.open(store=tomogram_wrapper.zarr(), path="/", mode="r")
        high_res_tomogram_zyx = z["0"][:]
        annotations_zyx = np.load(
            os.path.join(args.annotations_dir, f"{run.name}.npy")
        )

        for particle_name, particle_metadata in tqdm(
            particles.items(), desc="particle"
        ):
            picks = run.get_picks(
                particle_name, user_id="curation", session_id="0"
            )
            assert len(picks) == 1

            # Visualize each annotation independently
            for point in picks[0].points:
                # Visualize the yx slice at z level
                z_slice_index = round(point.location.z / args.voxel_spacing)
                tomogram_slice_yx = high_res_tomogram_zyx[z_slice_index]
                tomogram_slice_yx_img_gray = (
                    (tomogram_slice_yx - np.min(tomogram_slice_yx))
                    / (np.max(tomogram_slice_yx) - np.min(tomogram_slice_yx))
                ) * 255
                tomogram_slice_yx_img_gray = tomogram_slice_yx_img_gray.astype(
                    np.uint8
                )

                # Annotation yx slice, just set the white color corresponding to the annotation sphere
                ann_slice_yx_gray = annotations_zyx[z_slice_index]
                ann_slice_yx_gray[
                    ann_slice_yx_gray != particles[particle_name]["label"]
                ] = 0
                ann_slice_yx_gray[
                    ann_slice_yx_gray == particles[particle_name]["label"]
                ] = 255

                # Tomogram with annotation overlay
                tomogram_slice_yx_img_bgr = cv2.cvtColor(
                    tomogram_slice_yx_img_gray, cv2.COLOR_GRAY2BGR
                )
                ann_slice_yx_bgr = cv2.cvtColor(
                    ann_slice_yx_gray, cv2.COLOR_GRAY2BGR
                )
                # Set the color corresponding to the particle type
                ann_slice_yx_bgr = np.where(
                    ann_slice_yx_bgr == [255, 255, 255],
                    particle_metadata["color"][:3],
                    ann_slice_yx_bgr,
                ).astype(np.uint8)
                tomogram_slice_yx_img_ann = cv2.addWeighted(
                    tomogram_slice_yx_img_bgr, 0.5, ann_slice_yx_bgr, 0.15, 0
                )

                window_name_1 = f"tomogram_yx_z={z_slice_index}"
                cv2.imshow(window_name_1, tomogram_slice_yx_img_gray)
                window_name_2 = (
                    f"annotation_{particle_name}_yx_z={z_slice_index}"
                )
                cv2.imshow(window_name_2, ann_slice_yx_gray)
                window_name_3 = f"tomogram_with_annotation_{particle_name}_yx_z={z_slice_index}"
                cv2.imshow(window_name_3, tomogram_slice_yx_img_ann)
                cv2.waitKey(0)
                cv2.destroyWindow(window_name_1)
                cv2.destroyWindow(window_name_2)
                cv2.destroyWindow(window_name_3)


if __name__ == "__main__":
    main()
