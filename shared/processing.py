from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import zarr
from copick.impl.filesystem import CopickRunFSSpec
from copick.models import CopickPoint
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from skimage.morphology import ball
from sklearn.metrics import pairwise_distances


def gaussian_kernel(size: Tuple[int, int, int], sigma: int, device: str):
    """Generate a 3D Gaussian kernel."""
    kernel = np.fromfunction(
        lambda x, y, z: (1 / (2 * np.pi * sigma**2))
        * np.exp(
            -(
                (x - (size[0] - 1) / 2) ** 2
                + (y - (size[1] - 1) / 2) ** 2
                + (z - (size[2] - 1) / 2) ** 2
            )
            / (2 * sigma**2)
        ),
        size,
    )
    return (
        torch.tensor(kernel).float().unsqueeze(0).unsqueeze(0).to(device)
    )  # Add batch and channel dimensions


def create_hessian_particle_mask(
    tomogram: torch.Tensor, sigma: int, device: str
):
    """
    Generate a binary mask for dark, blob-like particles in a cryo-ET tomogram
    using Hessian-based filtering with PyTorch.

    Args:
        tomogram: The input 3D tomogram (C, H, W).
        sigma: The standard deviation for Gaussian smoothing.

    Returns:
        torch.Tensor: Binary mask highlighting dark blob-like areas in the tomogram.
    """
    kernel_size = (5, 5, 5)
    gaussian_k = gaussian_kernel(kernel_size, sigma, device=device)

    tomogram_smoothed = F.conv3d(
        tomogram.unsqueeze(0).unsqueeze(0), gaussian_k, padding=2
    ).squeeze()

    # Compute Hessian components
    hessian_xx = F.conv3d(
        tomogram_smoothed.unsqueeze(0).unsqueeze(0), gaussian_k, padding=2
    )
    hessian_yy = F.conv3d(
        tomogram_smoothed.unsqueeze(0).unsqueeze(0), gaussian_k, padding=2
    )
    hessian_xy = F.conv3d(
        tomogram_smoothed.unsqueeze(0).unsqueeze(0), gaussian_k, padding=2
    )

    hessian_response = (
        hessian_xx + hessian_yy + hessian_xy
    )  # Simplified combination
    binary_mask = hessian_response < 0  # Adjust threshold based on your needs

    return binary_mask.squeeze().byte()


def erode_dilate_mask(
    mask: torch.Tensor, radius: int, device: str
) -> torch.Tensor:
    """
    Perform binary erosion and dilation on a binary mask using a spherical structuring element.

    Args:
        mask: input binary mask
        radius: radius of the spherical structuring element
        device: device to use

    Returns:
        dilated mask after erosion and dilation operations
    """
    # Create a spherical structuring element
    struct_elem = ball(radius)
    struct_elem_tensor = (
        torch.tensor(struct_elem, dtype=torch.float32, device=device)
        .unsqueeze(0)
        .unsqueeze(0)
    )

    # Reshape mask for conv3d
    mask_reshaped = (
        mask.unsqueeze(0).unsqueeze(0).float()
    )  # Shape (1, 1, D, H, W)

    # Calculate padding size - ensure it's an integer
    pad_size = int(radius // 2)

    # Debug: Print shapes
    print(f"Mask shape for erosion: {mask_reshaped.shape}")
    print(f"Structuring element shape: {struct_elem_tensor.shape}")
    print(f"Padding size: {pad_size}")

    # Erosion: Use a negative structuring element for max pooling
    # Convert padding to the expected format (left, right, top, bottom, front, back)
    # Ensure all values are integers
    pad_3d = (
        int(pad_size),
        int(pad_size),
        int(pad_size),
        int(pad_size),
        int(pad_size),
        int(pad_size),
    )

    mask_padded = F.pad(mask_reshaped, pad_3d, mode="constant", value=1)
    eroded = -F.conv3d(
        -mask_padded,
        struct_elem_tensor,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    )
    eroded = (eroded >= struct_elem_tensor.sum()).squeeze().byte()

    # Dilation
    mask_padded = F.pad(
        eroded.unsqueeze(0).unsqueeze(0).float(),
        pad_3d,
        mode="constant",
        value=0,
    )
    dilated = F.conv3d(
        mask_padded,
        struct_elem_tensor,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    )
    dilated = (dilated > 0).squeeze().byte()

    return dilated


def distance_transform(mask: torch.Tensor, device: str) -> torch.Tensor:
    """
    Compute the distance transform using a simple distance transform approach.

    Args:
        mask: binary mask tensor

    Returns:
        Distance transform result
    """
    # Ensure mask is boolean, then convert to float for distance calculation
    mask = mask.bool()
    # Invert the mask (using logical not instead of bitwise not)
    inverted_mask = (~mask).float()

    # Add batch and channel dimensions
    inverted_mask = inverted_mask.unsqueeze(0).unsqueeze(0)

    # Create kernel on the correct device
    kernel = torch.ones(1, 1, 3, 3, 3, device=device)

    # Compute distance transform using convolution
    distance = F.conv3d(inverted_mask, kernel, padding=1)

    return distance.squeeze()


def local_maxima(distance: torch.Tensor, radius: int) -> torch.Tensor:
    """
    Detect local maxima in the distance transform.

    Args:
        distance: Distance transform tensor
        radius: Radius for local maxima detection

    Returns:
        torch.Tensor: Binary mask of local maxima
    """

    # Add batch dimension for max_pool3d
    distance = distance.unsqueeze(0)

    # Create kernel size tuple (must be odd numbers)
    kernel_size = (2 * radius + 1, 2 * radius + 1, 2 * radius + 1)

    # Compute local maxima
    maxpool = F.max_pool3d(
        distance, kernel_size=kernel_size, stride=1, padding=radius
    )

    # Compare with original distance to find local maxima
    local_max = distance == maxpool

    return local_max.squeeze()


def get_tomogram_data(
    run: CopickRunFSSpec,
    voxel_spacing: float,
    radius: float,
    tomo_type: str,
    resolution_threshold: int,
    device: str,
) -> Tuple[torch.Tensor, int, int]:
    """
    Get tomogram data at appropriate resolution based on particle radius.

    Args:
        run: Run object
        voxel_spacing: Base voxel spacing
        radius: Particle radius
        tomo_type: tomogram type to get
        resolution_threshold: resolution threshold to choose high or medium resolution
        device: device to use

    Returns:
        tuple: (tomogram tensor, effective_voxel_spacing, scale_factor)
    """
    tomogram_wrapper = run.get_voxel_spacing(voxel_spacing).get_tomogram(
        tomo_type
    )
    z = zarr.open(store=tomogram_wrapper.zarr(), path="/", mode="r")

    if radius <= resolution_threshold:
        # Use highest resolution
        tomogram = z["0"][:]
        effective_voxel_spacing = voxel_spacing
        scale_factor = 1
    else:
        # Use medium resolution
        tomogram = z["1"][:]
        effective_voxel_spacing = (
            voxel_spacing * 2
        )  # Scale factor is 2 for level 1
        scale_factor = 2

    return (
        torch.tensor(tomogram).to(device),
        effective_voxel_spacing,
        scale_factor,
    )


def remove_repeated_picks_v2(coordinates, distanceThreshold, pixelSize=1):
    # Calculate the distance matrix for the 3D coordinates
    dist_matrix = distance.cdist(
        coordinates[:, :3] / pixelSize, coordinates[:, :3] / pixelSize
    )

    # Create a linkage matrix using single linkage method
    Z = linkage(dist_matrix, method="complete")

    # Form flat clusters with a distance threshold to determine groups
    clusters = fcluster(Z, t=distanceThreshold, criterion="distance")

    # Initialize an array to store the average of each group
    unique_coordinates = np.zeros((max(clusters), coordinates.shape[1]))

    # Calculate the mean for each cluster
    for i in range(1, max(clusters) + 1):
        unique_coordinates[i - 1] = np.mean(coordinates[clusters == i], axis=0)

    return unique_coordinates


def remove_repeated_picks(coordinates, distanceThreshold, pixelSize=1):
    # Compute pairwise distances
    dist_matrix = pairwise_distances(coordinates)

    # Use hierarchical clustering to group close points
    distanceThreshold = distanceThreshold**2
    linkage_matrix = linkage(dist_matrix, method="single", metric="euclidean")
    labels = fcluster(linkage_matrix, t=distanceThreshold, criterion="distance")

    # Calculate the average coordinates for each group
    unique_labels = np.unique(labels)
    filteredCoordinates = np.array(
        [coordinates[labels == label].mean(axis=0) for label in unique_labels]
    )

    return filteredCoordinates


def calc_distances_matrix(points: List[CopickPoint]):
    points_coordinates = [
        np.array([point.location.x, point.location.y, point.location.z])
        for point in points
    ]
    return cdist(points_coordinates, points_coordinates, "euclidean")
