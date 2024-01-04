import numpy as np
from scipy.spatial.distance import euclidean
from typing import Tuple
from pathlib import Path
import pandas as pd


def target_registration_error(
    pts_fixed: np.ndarray, pts_moving: np.ndarray, voxel_size: Tuple[float]
) -> Tuple[float]:
    """Computes the mean and standard deviation for target registration error in mm between two arrays of sorted points.

    Args:
        pts_fixed (np.ndarray): fixed points
        pts_moving (np.ndarray): moving points
        voxel_size (Tuple[float]): the voxel size of the image

    Returns:
        Tuple[float]: Mean Error, Standar Error
    """
    voxel_size = np.array(voxel_size)[None, :]
    pts_fixed = pts_fixed * voxel_size
    pts_moving = pts_moving * voxel_size
    distances = [euclidean(pt1, pt2) for pt1, pt2 in zip(pts_fixed, pts_moving)]
    return np.around(np.mean(distances), 2), np.around(np.std(distances), 2)

def get_landmarks_array_from_txt_file(out_filepath: Path):
    """Parses the resulting txt from elastix to an array of landmark points [poits, [x,y,z]]

    Args:
        out_filepath (Path): save directory

    Returns:
        np.array: array of the ladmarks to use in TRE
    """
    landmarks = pd.read_csv(out_filepath, header=None, sep='\t |\t', engine='python')
    landmarks.columns = [
        'point', 'idx', 'input_index', 'input_point', 'ouput_index', 'ouput_point', 'def']
    landmarks = [lm[-4:-1] for lm in np.asarray(landmarks.ouput_index.str.split(' '))]
    landmarks = np.asarray(landmarks).astype('int')
    return landmarks