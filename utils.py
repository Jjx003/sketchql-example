from typing import List, Union
import math

import numpy as np
import torch


def normalize_flatten_data_centroid(centroid: np.ndarray, max_len: int = 128, max_obj: int = 4) -> np.ndarray:
    """
    Normalize and flatten centroid trajectory data for model input.
    
    This function performs several preprocessing steps:
    1. Downsample trajectories to a fixed length (max_len)
    2. Normalize coordinates using z-score normalization
    3. Pad to a fixed number of objects (max_obj)
    4. Flatten the data into a 2D array suitable for the transformer model
    
    Args:
        centroid (np.ndarray): Input centroid data of shape (n_objects, n_frames, 2)
                               where the last dimension contains (x, y) coordinates
        max_len (int, optional): Maximum sequence length after downsampling. 
                                Defaults to 128.
        max_obj (int, optional): Maximum number of objects to pad to. 
                                Defaults to 4.
    
    Returns:
        np.ndarray: Flattened and normalized data of shape (max_len, max_obj * 2)
                    where coordinates are concatenated as [obj1_x, obj1_y, obj2_x, obj2_y, ...]
    
    Example:
        >>> centroids = np.random.rand(2, 50, 2)  # 2 objects, 50 frames, 2 coords
        >>> normalized = normalize_flatten_data_centroid(centroids)
        >>> print(normalized.shape)  # (128, 8)
    
    Note:
        - If input has fewer than max_obj objects, the remaining slots are zero-padded
        - If input has more than max_len frames, it's downsampled using linear interpolation
        - Normalization is applied per coordinate dimension across all objects and frames
    """
    # Downsample to max_len frames
    ratio = centroid.shape[1] / max_len
    keep_idx = np.array([int(i * ratio) for i in range(max_len)])
    centroid = centroid[:, keep_idx, :]
    
    # Z-score normalization
    x_bar = np.mean(centroid, axis=(0, 1))
    x_std = np.std(centroid, axis=(0, 1))
    centroid = (centroid - x_bar) / (x_std + 1e-6)

    # Pad to max_obj objects
    centroid_pad = np.zeros(shape=(max_obj, centroid.shape[1], centroid.shape[2]))
    centroid_pad[:centroid.shape[0], :, :] = centroid
    
    # Flatten: concatenate all objects' trajectories
    centroid_pad_flat = [centroid_pad[i_obj] for i_obj in range(max_obj)]
    centroid_pad_flat = np.concatenate(centroid_pad_flat, axis=1)
    
    return centroid_pad_flat


def relative_to_center(centroids: np.ndarray) -> np.ndarray:
    """
    Center trajectories by subtracting the mean position.
    
    This function normalizes trajectories by making them relative to their center
    of mass, which can help with translation-invariant matching.
    
    Args:
        centroids (np.ndarray): Input centroid data of shape (n_objects, n_frames, 2)
    
    Returns:
        np.ndarray: Centered centroid data of the same shape as input
    
    Example:
        >>> centroids = np.array([[[100, 200], [110, 210]], [[300, 400], [310, 410]]])
        >>> centered = relative_to_center(centroids)
        >>> print(centered.mean(axis=(0, 1)))  # [0, 0] (approximately)
    """
    mean = np.mean(centroids, axis=0)
    return centroids - mean


def centroids2model_input(centroids: np.ndarray) -> torch.Tensor:
    """
    Convert centroid trajectories to PyTorch tensor format for model input.
    
    This is a convenience function that combines normalization/flattening with
    tensor conversion. It's the main entry point for preparing centroid data
    for the encoder model.
    
    Args:
        centroids (np.ndarray): Input centroid data of shape (n_objects, n_frames, 2)
    
    Returns:
        torch.Tensor: Model-ready tensor of shape (max_len, max_obj * 2) with dtype float32
    
    Example:
        >>> centroids = np.random.rand(2, 50, 2)
        >>> model_input = centroids2model_input(centroids)
        >>> print(model_input.shape)  # torch.Size([128, 8])
        >>> print(model_input.dtype)  # torch.float32
    """
    centroid_size_flat = normalize_flatten_data_centroid(centroids)
    centroid_size_flat = torch.from_numpy(centroid_size_flat).float()
    return centroid_size_flat


def convert2centroids(bbox_list: List[List[List[Union[int, float]]]]) -> np.ndarray:
    """
    Convert bounding box sequences to centroid sequences.
    
    This function extracts the center point (centroid) of each bounding box
    and ensures all sequences have the same length by truncating to the shortest.
    
    Args:
        bbox_list (List[List[List[Union[int, float]]]]): List of bounding box sequences.
            Each sequence is a list of bounding boxes, where each bounding box is
            [x1, y1, x2, y2] representing top-left and bottom-right corners.
            Example: [[[100, 200, 150, 250], [105, 205, 155, 255], ...],  # Object 1
                     [[300, 400, 350, 450], [305, 405, 355, 455], ...]]  # Object 2
    
    Returns:
        np.ndarray: Centroid data of shape (n_objects, min_sequence_length, 2)
                    where the last dimension contains (x_center, y_center) coordinates
    
    Example:
        >>> bboxes = [[[100, 200, 150, 250], [105, 205, 155, 255]],
        ...           [[300, 400, 350, 450], [305, 405, 355, 455]]]
        >>> centroids = convert2centroids(bboxes)
        >>> print(centroids.shape)  # (2, 2, 2)
        >>> print(centroids[0, 0])  # [125.0, 225.0] - center of first bbox
    
    Note:
        - All sequences are truncated to the length of the shortest sequence
        - Centroid calculation: x_center = (x1 + x2) / 2, y_center = (y1 + y2) / 2
        - Input validation is minimal; ensure bbox format is correct
    """
    # Find minimum sequence length and truncate all sequences
    min_len = min([len(it) for it in bbox_list])
    bbox_list = [it[:min_len] for it in bbox_list]
    
    # Convert each bounding box sequence to centroids
    centroid_list = []
    for boxes in bbox_list:
        boxes = np.array(boxes)
        centroids = np.zeros(shape=(boxes.shape[0], 2))
        centroids[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2  # x_center
        centroids[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2  # y_center
        centroid_list.append(centroids)
    
    centroids = np.array(centroid_list)
    return centroids

def rotate_by_orgin(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    
    This function performs 2D rotation using the standard rotation matrix formula.
    The rotation is applied around the specified origin point, not the coordinate origin.
    
    Args:
        origin (tuple): Origin point (x, y) around which to rotate
        point (tuple): Point (x, y) to be rotated
        angle (float): Rotation angle in radians (positive for counterclockwise)
    
    Returns:
        tuple: Rotated point coordinates (qx, qy)
    
    Example:
        >>> origin = (0, 0)
        >>> point = (1, 0)
        >>> rotated = rotate_by_orgin(origin, point, math.pi/2)
        >>> print(rotated)  # (0.0, 1.0)
    
    Note:
        - Angle should be given in radians, not degrees
        - Counterclockwise rotation is positive, clockwise is negative
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def rotate_centroids(centroids, angle, origin):
    """
    Rotate all centroids by a given angle around a specified origin.
    
    This function applies the same rotation to all centroid points in the array,
    rotating them around the specified origin point.
    
    Args:
        centroids (np.ndarray): Array of centroid points of shape (n_points, 2)
        angle (float): Rotation angle in radians (positive for counterclockwise)
        origin (tuple): Origin point (x, y) around which to rotate
    
    Returns:
        np.ndarray: Rotated centroids of the same shape as input
    
    Example:
        >>> centroids = np.array([[1, 0], [0, 1], [-1, 0]])
        >>> origin = (0, 0)
        >>> rotated = rotate_centroids(centroids, math.pi/2, origin)
        >>> print(rotated)  # [[0, 1], [-1, 0], [0, -1]]
    
    Note:
        - All points are rotated by the same angle around the same origin
        - Input centroids should be 2D array with shape (n_points, 2)
    """
    new_centroids = []
    for i in range(len(centroids)):
        p = rotate_by_orgin(origin, (centroids[i][0], centroids[i][1]), angle)
        new_centroids.append(p)
    return np.array(new_centroids)

def rotate_clip_centroids(centroids, degree=None):
    """
    Rotate all centroid trajectories by a random or specified angle.
    
    This function rotates all object trajectories around their collective center of mass.
    It's useful for data augmentation by applying random rotations to training data.
    
    Args:
        centroids (np.ndarray): Centroid data of shape (n_objects, n_frames, 2)
        degree (int, optional): Rotation angle in degrees. If None, uses random angle 0-360.
    
    Returns:
        np.ndarray: Rotated centroids of the same shape as input
    
    Example:
        >>> centroids = np.random.rand(2, 10, 2)  # 2 objects, 10 frames
        >>> rotated = rotate_clip_centroids(centroids, degree=90)
        >>> print(rotated.shape)  # (2, 10, 2)
    
    Note:
        - Rotation origin is calculated as the mean of all centroids across all objects and frames
        - If degree is None, a random angle between 0-360 degrees is used
        - Angle is converted from degrees to radians internally
    """
    import random
    if degree is None:
        degree = random.randint(0,360)
    angle = (degree-180)/180*math.pi
    origin = np.mean(centroids, axis=(0, 1))
    new_centroids = []
    for i in range(centroids.shape[0]):
        new_centroids.append(rotate_centroids(centroids[i], angle, origin))
    return np.array(new_centroids)