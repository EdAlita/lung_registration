# Lung MRI Registration Project

## Table of Contents

1. [**Introduction**](#introduction)
   - Overview and evaluation method using Target Registration Error (TRE).
2. [**Objectives**](#objectives)
   - Development, automation, and validation of the lung image registration process.
3. [**Folders**](#folders)
   - Descriptions of various folders like Noteebooks, Data, Parameters, etc.
4. [**Detailed Description of Key Folders**](#detailed-description-of-key-folders)
   - Specifics about `Utils` and `Notebooks` folders.
5. [**Installation and Usage**](#installation-and-usage)
   - Software requirements, installation guide, and usage instructions.
6. [**Results**](#results)
   - Summary of project achievements and TRE results visualization.
7. [**Authors**](#authors)
   - Contributions and profiles of the project team.
8. [**Achievements**](#achievements)
   - Awards and recognitions in relevant challenges.
9. [**License**](#license)
   - Licensing information of the project.


## Introduction
This project aims to address the challenge of registering lung CT images between exhalation and inhalation phases. By aligning these images, the project seeks to facilitate detailed observation of lung tissue movement and changes. The effectiveness of image registration is evaluated using the Target Registration Error (TRE).

## Objectives
1. Develop an algorithm for registering CT lung images between exhalation and inhalation phases.
2. Automate the registration and TRE estimation process from 3D lung volumes.
3. Validate the registration results through rigorous analysis.

## Folders
- [**Noteebooks**](noteebooks): Notebooks of the implementation of a registration.
- [**Data**](data): raw data and landmarks information.
- [**Parameters**](parameters): parameters files used on the registration with elastik.
- [**Experiments**](experiments): Numeral results of our testing in json files
- [**Result Data**](result_data): The data result from the preprocess and load data.
- [**Utils**](utils): Functions develop to solve this project.

### Utils
- `elastic_helpers.py`: Facilitates the use of Elastix for image registration, simplifying complex processes.
- `metrics.py`: Provides functions for calculating TRE and other metrics, essential for assessing registration accuracy.
- `dataset.py`: Handles the loading, preprocessing, and management of MRI data.

### Notebooks
- `01_dataset.ipynb`: Demonstrates how to load and preprocess the dataset.
- `02_elastix.ipynb`: Details the implementation of the Elastix registration.
- `03_ans.ipynb`: Implementation of [Advance Normalization Tool](https://github.com/ANTsX/ANTsPy). **WIP**
- `04_analyze_results.ipynb`: For analyzing and visualizing the results of the registration.

## Installation and Usage
- **Requirements**: Python along with additional dependencies listed in a `requirements.txt` file.
- **Installation**: Install the necessary Python packages using `pip install -r requirements.txt`.
- **Usage**: Follow the Jupyter notebooks in sequence to understand the data handling, registration process, and result analysis.


## Results
The project successfully achieved automated registration of lung MRI images, with accuracy validated by TRE calculations. Detailed results are presented in the [Jupyter notebooks](noteebooks/04_analyze_results.ipynb).

### TRE Results from experiments

![alt text](https://github.com/EdAlita/lung_registration/blob/main/images/TRE-unmaks.png?raw=true)

![alt text](https://github.com/EdAlita/lung_registration/blob/main/images/TRE-maks.png?raw=true)

![alt text](https://github.com/EdAlita/lung_registration/blob/main/images/ants.png?raw=true)

## Authors
- [Carmen Colin](https://github.com/CarmenColinTen)
- [Edwing Ulin](https://github.com/EdAlita)

## Achivements

- 3rd place on the Non rigid 3D Lung CT registration challenge.
    - 1st place of the elastik algorithm

## License
This project is licensed under the Creative Common Lincense - see the [LICENSE.md](LICENSE) file for details.
