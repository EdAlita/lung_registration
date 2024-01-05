import SimpleITK as sitk
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import logging
import json
import tempfile
import shutil
from typing import Tuple
from rich.progress import track
import matplotlib.pyplot as plt
import skimage
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border, mark_boundaries
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.misc

def get_segmented_lungs(
    raw_im: np.ndarray, plot: bool = False
    ) -> np.ndarray:
    """Generates a 2D mask of an image representing segmented lungs.

    Args:
        raw_im (np.ndarray): Numpy array representing a lung slice.
        plot (bool, optional): Whether to plot all the intermediate steps. Defaults to False.

    Returns:
        np.ndarray: Binary mask of segmented lungs.
    """
    if not isinstance(raw_im, np.ndarray):
        raise ValueError("Input image must be a numpy array.")

    try:
        im = raw_im.copy()
        plots = None
        if plot:
            f, plots = plt.subplots(8, 1, figsize=(5, 40))

        def plot_step(data, index, cmap=plt.cm.bone):
            if plot:
                plots[index].axis('off')
                plots[index].imshow(data, cmap=cmap)

        binary = im < 400
        plot_step(binary, 0)

        cleared = clear_border(binary)
        plot_step(cleared, 1)

        label_image = label(cleared)
        plot_step(label_image, 2)

        areas = [r.area for r in regionprops(label_image)]
        areas.sort()
        if len(areas) > 2:
            for region in regionprops(label_image):
                if region.area < areas[-2]:
                    for coordinates in region.coords:                
                        label_image[coordinates[0], coordinates[1]] = 0

        binary = label_image > 0
        plot_step(binary, 3)

        selem = disk(2)
        binary = binary_erosion(binary, selem)
        plot_step(binary, 4)

        selem = disk(10)
        binary = binary_closing(binary, selem)
        plot_step(binary, 5)

        edges = roberts(binary)
        binary = ndi.binary_fill_holes(edges)
        plot_step(binary, 6)

        get_high_vals = binary == 0
        im[get_high_vals] = 0
        plot_step(im, 7)

        return binary
    except Exception as e:
        raise RuntimeError(f"Error during lung segmentation: {e}")

def get_segmented_lungs_3d(
    image_volume: np.ndarray, output_path: str, spacing: list = [1.0, 1.0, 1.0]
    ) -> np.ndarray:
    """
    Applies 2D lung segmentation to a 3D image volume.

    Args:
        image_volume (np.ndarray): 3D numpy array representing the image volume.
        output_path (str): Path to write the segmented image.
        spacing (list, optional): Spacing information for the image. Defaults to (1.0, 1.0, 1.0).

    Returns:
        np.ndarray: 3D binary mask for debug purposes.
    """
    if not isinstance(image_volume, np.ndarray) or image_volume.ndim != 3:
        raise ValueError("image_volume must be a 3D numpy array.")
    if not isinstance(spacing, list) or len(spacing) != 3:
        raise ValueError("spacing must be a tuple of three elements.")

    try:
        binary_masks = np.zeros_like(image_volume)

        for i in range(image_volume.shape[0]):
            slice_image = image_volume[i, :, :]
            binary_mask = get_segmented_lungs(slice_image)
            binary_masks[i, :, :] = binary_mask

        binary_mask_sitk = sitk.GetImageFromArray(binary_masks.astype(np.uint8))
        binary_mask_sitk.SetSpacing(spacing)
        sitk.WriteImage(binary_mask_sitk, output_path)

        return binary_masks
    except Exception as e:
        raise RuntimeError(f"Error during 3D lung segmentation: {e}")

def read_raw_sitk(
    binary_file_path: Path, image_size: Tuple[int], sitk_pixel_type: int = sitk.sitkInt16,
    image_spacing: Tuple[float, float, float] = None, image_origin: Tuple[float, float, float] = None, big_endian: bool = False
) -> sitk.Image:
    """
    Reads a image raw data to create a SimpleITK Image.

    Args:
        binary_file_path (Path): Location of the binary file.
        image_size (Tuple[int]): Size of the image to produce (e.g., (width, height) or (width, height, depth)).
        sitk_pixel_type (int, optional): Pixel type of the image. Defaults to sitk.sitkInt16.
        image_spacing (Tuple[float], optional): Spacing of the image. Defaults to None.
        image_origin (Tuple[float], optional): Origin of the image. Defaults to None.
        big_endian (bool, optional): Set to True for big endian data. Defaults to False.

    Returns:
        sitk.Image: The resulting SimpleITK image.
    """
    # Input validation
    if not binary_file_path.is_file():
        raise FileNotFoundError(f"The file {binary_file_path} does not exist.")
    if len(image_size) not in [2, 3]:
        raise ValueError("image_size must be a 2D or 3D tuple.")
    # Add validations for image_spacing and image_origin if needed

    pixel_dict = {
        sitk.sitkUInt8: "MET_UCHAR",
        sitk.sitkInt8: "MET_CHAR",
        sitk.sitkUInt16: "MET_USHORT",
        sitk.sitkInt16: "MET_SHORT",
        sitk.sitkUInt32: "MET_UINT",
        sitk.sitkInt32: "MET_INT",
        sitk.sitkUInt64: "MET_ULONG_LONG",
        sitk.sitkInt64: "MET_LONG_LONG",
        sitk.sitkFloat32: "MET_FLOAT",
        sitk.sitkFloat64: "MET_DOUBLE",
    }
    direction_cosine = [
        "1 0 0 1",
        "1 0 0 0 1 0 0 0 1",
        "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1",
    ]

    dim = len(image_size)

    element_spacing = " ".join(["1"] * dim)
    if image_spacing is not None:
        element_spacing = " ".join([str(v) for v in image_spacing])

    img_origin = " ".join(["0"] * dim)
    if image_origin is not None:
        img_origin = " ".join([str(v) for v in image_origin])

    header = [
        ("ObjectType = Image\n").encode(),
        (f"NDims = {dim}\n").encode(),
        (f'DimSize = {" ".join([str(v) for v in image_size])}\n').encode(),
        (f"ElementSpacing = {element_spacing}\n").encode(),
        (f"Offset = {img_origin}\n").encode(),
        (f"TransformMatrix = {direction_cosine[dim - 2]}\n").encode(),
        (f"ElementType = {pixel_dict[sitk_pixel_type]}\n").encode(),
        ("BinaryData = True\n").encode(),
        ("BinaryDataByteOrderMSB = " + str(big_endian) + "\n").encode(),
        (f"ElementDataFile = {binary_file_path.resolve()}\n").encode(),
    ]

    # Using context manager for temporary file handling
    try:
        with tempfile.NamedTemporaryFile(suffix=".mhd", delete=False) as fp:
            fp.writelines(header)
            temp_file_name = fp.name

        img = sitk.ReadImage(temp_file_name)
    finally:
        Path(temp_file_name).unlink()

    return img
def generate_landmark_transformix_header(
    case_out_path: Path, lm_path: Path
):
    """Puts the header for using with transformix

    Args:
        case_out_path (Path): case path 
        lm_path (Path): out path for the document
    """
    txt_out_file = case_out_path / f'{lm_path.stem}.txt'
    shutil.copy(str(lm_path), str(txt_out_file))
    with open(txt_out_file, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write('index' + '\n' + '300' + '\n' + content)

def generate_csv_from_landmarks(
    case_out_path: Path,lm_path: Path
) -> Path:
    """Generates the csv from the points given

    Args:
        case_out_path (Path): case path 
        lm_path (Path): out path for the document
    """
    landmarks = pd.read_csv(
        lm_path, header=None, sep='\t |\t', engine='python').astype('int')
    lm_pts_out_path = case_out_path / f'{lm_path.stem}.csv'
    landmarks.to_csv(lm_pts_out_path, index=False, header=False)

    return lm_pts_out_path

def parse_raw_images(data_path: Path, out_path: Path
):
    """Produce the data structure from a datapath of raw images and points

    Args:
        data_path (Path): location of the data to convert
        out_path (Path): space to save this data

    Returns:
        dataframe: all the locations and information of the images
    """
    if not data_path.is_dir():
        raise NotADirectoryError(f"The data path {data_path} is not a directory.")
    if not out_path.is_dir():
        raise NotADirectoryError(f"The output path {out_path} is not a directory.")

    with open(str(data_path.parent / 'dir_lab_copd_metadata.json'), 'r') as json_file:
        dirlab_meta = json.load(json_file)

    df = []
    for case_path in sorted(data_path.iterdir()):
        # Define paths
        case = case_path.name
        ilm_path = case_path / f'{case}_300_iBH_xyz_r1.txt'
        i_img_path = case_path / f'{case}_iBHCT.img'
        elm_path = case_path / f'{case}_300_eBH_xyz_r1.txt'
        e_img_path = case_path / f'{case}_eBHCT.img'

        case_out_path = out_path / case
        case_out_path.mkdir(exist_ok=True, parents=True)

        # Get metadata:
        meta = dirlab_meta[case]

        # Parse raw image and parse landmarks
        img_out_paths, mask_out_paths, lm_pts_out_paths = [], [], []
        for img_path, lm_path in zip([i_img_path, e_img_path], [ilm_path, elm_path]):
            img = read_raw_sitk(
                img_path, meta['size'], sitk.sitkInt16, meta['spacing'])
            img_out_path = case_out_path / f'{img_path.stem}.nii.gz'
            sitk.WriteImage(img, str(img_out_path))

            generate_landmark_transformix_header(case_out_path=case_out_path,lm_path=lm_path)

            lm_pts_out_path = generate_csv_from_landmarks(case_out_path=case_out_path,lm_path=lm_path)

            mask_out_path = case_out_path / f'{img_path.stem}_masks.nii.gz'
            img = sitk.GetArrayFromImage(sitk.ReadImage(str(img_out_path)))
            _ = get_segmented_lungs_3d(img,output_path=str(mask_out_path),spacing=meta['spacing'])

            img_out_paths.append('/'.join(str(img_out_path).split('/')[-4:]))
            lm_pts_out_paths.append('/'.join(str(lm_pts_out_path).split('/')[-4:]))
            mask_out_paths.append('/'.join(str(mask_out_path).split('/')[-4:]))

        metrics_keys = [
            'disp_mean', 'disp_std', 
            ]
        row = img_out_paths  + lm_pts_out_paths + mask_out_paths
        row = row  + list(meta['size']) + list(meta['spacing']) + [case]
        row = row + [meta[key] for key in metrics_keys]
        df.append(row)

    columns = [
        'i_img_path', 'e_img_path','i_landmark_pts', 'e_landmark_pts','i_mask_path','e_mask_path'
        , 'size_x', 'size_y', 'size_z', 'space_x', 'space_y', 'space_z', 'case'
    ]

    columns = columns + metrics_keys
    df = pd.DataFrame(df, columns=columns)
    
    return df

def plot_random_layers(nifti_file1, nifti_file2, case):
    """Plots a random layer from each of two 3D NIfTI files.

    Args:
    nifti_file1 (str): Path to the first NIfTI file.
    nifti_file2 (str): Path to the second NIfTI file.
    """
    img1 = sitk.ReadImage(nifti_file1)
    img2 = sitk.ReadImage(nifti_file2)

    data1 = sitk.GetArrayFromImage(img1)
    data2 = sitk.GetArrayFromImage(img2)

    if len(data1.shape) != 3 or len(data2.shape) != 3:
        raise ValueError("One or both NIfTI files do not contain 3D data.")

    print(f'Size: {data1.shape}')
    layer1 = np.random.randint(data1.shape[0])

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(data1[layer1,: ,: ], cmap='gray')
    axes[0].set_title(f'{case}_inhale')
    axes[0].axis('off')
    axes[1].imshow(data2[layer1, :,:], cmap='gray')
    axes[1].set_title(f'{case}_exhale')
    axes[1].axis('off')

    plt.show()



