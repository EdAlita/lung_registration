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



def read_raw_sitk(
    binary_file_path: Path, image_size: Tuple[int], sitk_pixel_type: int = sitk.sitkInt16,
    image_spacing: Tuple[float] = None, image_origin: Tuple[float] = None, big_endian: bool = False
) -> sitk.Image:
    """ Reads a raw binary scalar image.
    Args:
        binary_file_path (Path): Raw, binary image file path.
        image_size (Tuple): Size of image (e.g. (512, 512, 121))
        sitk_pixel_type (int, optional): Pixel type of data.
            Defaults to sitk.sitkInt16.
        image_spacing (Tuple, optional): Image spacing, if none given assumed
            to be [1]*dim. Defaults to None.
        image_origin (Tuple, optional): image origin, if none given assumed to
            be [0]*dim. Defaults to None.
        big_endian (bool, optional): Byte order indicator, if True big endian, else
            little endian. Defaults to False.
    Returns:
        (sitk.Image): Loaded image.
    """
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
    fp = tempfile.NamedTemporaryFile(suffix=".mhd", delete=False)
    # Not using the tempfile with a context manager and auto-delete
    # because on windows we can't open the file a second time for ReadImage.
    fp.writelines(header)
    fp.close()

    img = sitk.ReadImage(str(fp.name))
    Path(fp.name).unlink()
    return img

def parse_raw_images(data_path: Path, out_path: Path):
    """
    Parses the raw images contained in data_path.
    Args:
        data_path (Path): path to the directory containing the raw cases
        out_path (Path): path to the directory where the parsed .nii versions
            will be saved
    Returns:
        (pd.DataFrame): dataframe of the metadata to be used in the CopdDataset class
    """
    with open(str(data_path.parent / 'dir_lab_copd_metadata.json'), 'r') as json_file:
        dirlab_meta = json.load(json_file)

    df = []
    for case_path in sorted(data_path.iterdir()):
        # Define paths
        case = case_path.name
        logging.info(f'Parsing case: {case}')
        ilm_path = case_path / f'{case}_300_iBH_xyz_r1.txt'
        i_img_path = case_path / f'{case}_iBHCT.img'
        elm_path = case_path / f'{case}_300_eBH_xyz_r1.txt'
        e_img_path = case_path / f'{case}_eBHCT.img'

        case_out_path = out_path / case
        case_out_path.mkdir(exist_ok=True, parents=True)

        # Get metadata:
        meta = dirlab_meta[case]

        # Parse raw image and parse landmarks
        img_out_paths, lm_out_paths, lm_pts_out_paths = [], [], []
        for img_path, lm_path in zip([i_img_path, e_img_path], [ilm_path, elm_path]):
            img = read_raw_sitk(
                img_path, meta['size'], sitk.sitkInt16, meta['spacing'])
            # flip vertical axis:
            img_out_path = case_out_path / f'{img_path.stem}.nii.gz'
            sitk.WriteImage(img, str(img_out_path))

            # Generate a copy of the landmarks that includes transformix header
            txt_out_file = case_out_path / f'{lm_path.stem}.txt'
            shutil.copy(str(lm_path), str(txt_out_file))
            with open(txt_out_file, 'r+') as f:
                content = f.read()
                f.seek(0, 0)
                f.write('index' + '\n' + '300' + '\n' + content)

            # Generate a csv version of the landmarks
            landmarks = pd.read_csv(
                lm_path, header=None, sep='\t |\t', engine='python').astype('int')
            lm_pts_out_path = case_out_path / f'{lm_path.stem}.csv'
            landmarks.to_csv(lm_pts_out_path, index=False, header=False)
            landmarks = landmarks.values
            """
            # Generate landmarks mask
            lm_mask = data_utils.generate_lm_mask(landmarks, meta['size'])
            lm_mask = np.moveaxis(lm_mask, [0, 1, 2], [2, 1, 0])

            lm_out_path = case_out_path / f'{img_path.stem}_lm.nii.gz'
            utils.save_img_from_array_using_referece(lm_mask, img, str(lm_out_path))
            """
            img_out_paths.append('/'.join(str(img_out_path).split('/')[-4:]))
            lm_pts_out_paths.append('/'.join(str(lm_pts_out_path).split('/')[-4:]))
            
        # Store the sample metadata
        metrics_keys = [
            'disp_mean', 'disp_std', 
            #'observers_mean', 'observers_std', 'lowest_mean', 'lowest_std'
            ]
        row = img_out_paths  + lm_pts_out_paths
        row = row  + list(meta['size']) + list(meta['spacing']) + [case]
        row = row + [meta[key] for key in metrics_keys]
        df.append(row)
    columns = [
        'i_img_path', 'e_img_path','i_landmark_pts', 'e_landmark_pts'
        , 'size_x', 'size_y', 'size_z', 'space_x', 'space_y', 'space_z', 'case'
    ]
    columns = columns + metrics_keys
    df = pd.DataFrame(df, columns=columns)
    
    return df

def plot_random_layers(nifti_file1, nifti_file2, case):
    """
    Plots a random layer from each of two 3D NIfTI files.

    Parameters:
    nifti_file1 (str): Path to the first NIfTI file.
    nifti_file2 (str): Path to the second NIfTI file.
    """

    # Load the NIfTI files
    img1 = sitk.ReadImage(nifti_file1)
    img2 = sitk.ReadImage(nifti_file2)

    # Convert the images to numpy arrays
    data1 = sitk.GetArrayFromImage(img1)
    data2 = sitk.GetArrayFromImage(img2)

    # Ensure the data is 3D
    if len(data1.shape) != 3 or len(data2.shape) != 3:
        raise ValueError("One or both NIfTI files do not contain 3D data.")

    # Choose a random layer from each file
    print(f'Size: {data1.shape}')
    layer1 = np.random.randint(data1.shape[0])

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(data1[layer1,: ,: ], cmap='gray')
    axes[0].set_title(f'{case}_inhale')
    axes[0].axis('off')
    axes[1].imshow(data2[layer1, :,:], cmap='gray')
    axes[1].set_title(f'{case}_exhale')
    axes[1].axis('off')

    plt.show()
