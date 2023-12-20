import numpy as np
from scipy.spatial.distance import euclidean
from typing import Tuple
from pathlib import Path
import pandas as pd


def target_registration_error(
    pts_i: np.ndarray, pts_e: np.ndarray, voxel_size: Tuple[float]
) -> Tuple[float]:
    """ Computes the mean and standard deviation for target registration error in mm
    between two arrays of sorted points.
    Warning: The points should have the same order in each array

    Args:
        pts_i (np.ndarray): _description_
        pts_e (np.ndarray): _description_
        voxel_size (Tuple[float]): _description_
    Returns:
        float: Target registration error mean
        float: Target registration error standard deviation
    """
    voxel_size = np.array(voxel_size)[None, :]
    pts_i = pts_i * voxel_size
    pts_e = pts_e * voxel_size
    distances = [euclidean(pt1, pt2) for pt1, pt2 in zip(pts_i, pts_e)]
    return np.around(np.mean(distances), 2), np.around(np.std(distances), 2)

def get_landmarks_array_from_txt_file(lm_out_filepath: Path):
    """Parses the resulting txt from elastix to an array of landmark points [poits, [x,y,z]]"""
    landmarks = pd.read_csv(lm_out_filepath, header=None, sep='\t |\t', engine='python')
    landmarks.columns = [
        'point', 'idx', 'input_index', 'input_point', 'ouput_index', 'ouput_point', 'def']
    landmarks = [lm[-4:-1] for lm in np.asarray(landmarks.ouput_index.str.split(' '))]
    landmarks = np.asarray(landmarks).astype('int')
    return landmarks