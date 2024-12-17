import argparse
import os

import copick
import cv2
import numpy as np
import zarr
from tqdm import tqdm


def normalize_min_max(input_gray):
    output_gray = (
        (input_gray - np.min(input_gray))
        / (np.max(input_gray) - np.min(input_gray))
    ) * 255
    output_gray = output_gray.astype(np.uint8)
    return output_gray


def set_ann_value(ann_slice_yx_gray, label):
    """
    Set white color where the specific label is set, black otherwise
    """
    ann_slice_yx_gray[ann_slice_yx_gray != label] = 0
    ann_slice_yx_gray[ann_slice_yx_gray == label] = 255


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
                # Center
                x_slice_idx = round(point.location.x / args.voxel_spacing)
                y_slice_idx = round(point.location.y / args.voxel_spacing)
                z_slice_idx = round(point.location.z / args.voxel_spacing)
                print(
                    f"{particle_name} at z, y, x: {z_slice_idx}, {y_slice_idx}, {x_slice_idx}"
                )

                # Visualize the yx slice at z level
                tomogram_slice_yx = high_res_tomogram_zyx[z_slice_idx]
                tomogram_slice_yx_img_gray = normalize_min_max(
                    tomogram_slice_yx
                )

                # Visualize the zx slice
                tomogram_slice_zx = high_res_tomogram_zyx[:, y_slice_idx, :]
                tomogram_slice_zx_img_gray = normalize_min_max(
                    tomogram_slice_zx
                )

                # Annotation yx slice at z
                ann_slice_yx_gray = annotations_zyx[z_slice_idx]
                set_ann_value(ann_slice_yx_gray, particle_metadata["label"])

                # Annotation zx slice at y
                ann_slice_zx_gray = annotations_zyx[:, y_slice_idx, :]
                set_ann_value(ann_slice_zx_gray, particle_metadata["label"])

                # Tomogram with annotation overlay at z
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

                window_name_1 = f"tomogram_yx_z={z_slice_idx}"
                cv2.imshow(window_name_1, tomogram_slice_yx_img_gray)
                window_name_2 = f"annotation_{particle_name}_yx_z={z_slice_idx}"
                cv2.imshow(window_name_2, ann_slice_yx_gray)
                window_name_3 = f"tomogram_with_annotation_{particle_name}_yx_z={z_slice_idx}"
                cv2.imshow(window_name_3, tomogram_slice_yx_img_ann)
                window_name_4 = f"tomogram_zx_y={y_slice_idx}"
                cv2.imshow(window_name_4, tomogram_slice_zx_img_gray)
                window_name_5 = f"annotation_{particle_name}_zx_y={y_slice_idx}"
                cv2.imshow(window_name_5, ann_slice_zx_gray)
                cv2.waitKey(0)
                cv2.destroyWindow(window_name_1)
                cv2.destroyWindow(window_name_2)
                cv2.destroyWindow(window_name_3)
                cv2.destroyWindow(window_name_4)
                cv2.destroyWindow(window_name_5)


if __name__ == "__main__":
    main()
