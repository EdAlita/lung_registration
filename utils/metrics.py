import numpy as np
from scipy.spatial.distance import euclidean
from typing import Tuple
from pathlib import Path
import pandas as pd


from scipy.spatial.distance import euclidean

def target_registration_error(
    pts_fixed: np.ndarray, pts_moving: np.ndarray, voxel_size: list[int, int, int]
) -> Tuple[float, float]:
    """Computes the mean and standard deviation for target registration error in mm.

    Args:
        pts_fixed (np.ndarray): Fixed points.
        pts_moving (np.ndarray): Moving points.
        voxel_size (Tuple[int, int, int]): The voxel size of the image.

    Returns:
        List[int, int]: Mean Error, Standard Error.
    """
    # Input validation and error handling
    if not (isinstance(pts_fixed, np.ndarray) and isinstance(pts_moving, np.ndarray)):
        raise ValueError("pts_fixed and pts_moving must be numpy arrays.")
    if pts_fixed.shape != pts_moving.shape:
        raise ValueError("pts_fixed and pts_moving must have the same shape.")
    if not isinstance(voxel_size, list) or len(voxel_size) != 3:
        raise ValueError("voxel_size must be a tuple of three three.")

    try:
        voxel_size = np.array(voxel_size)[None, :]
        pts_fixed_scaled = pts_fixed * voxel_size
        pts_moving_scaled = pts_moving * voxel_size
        distances = np.array([euclidean(pt1, pt2) for pt1, pt2 in zip(pts_fixed_scaled, pts_moving_scaled)])
        return np.mean(distances).round(2), np.std(distances).round(2)
    except Exception as e:
        raise RuntimeError(f"Error in calculating distances: {e}")


def get_landmarks_array_from_txt_file(lm_out_filepath: Path):
    """Parses the resulting txt from elastix to an array of landmark points [poits, [x,y,z]]"""
    landmarks = pd.read_csv(lm_out_filepath, header=None, sep='\t |\t', engine='python')
    landmarks.columns = [
        'point', 'idx', 'input_index', 'input_point', 'ouput_index', 'ouput_point', 'def']
    landmarks = [lm[-4:-1] for lm in np.asarray(landmarks.ouput_index.str.split(' '))]
    landmarks = np.asarray(landmarks).astype('int')
    return landmarks