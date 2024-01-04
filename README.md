# Lung MRI Registration Project

## Introduction
This project aims to address the challenge of registering lung MRI images between exhalation and inhalation phases. By aligning these images, the project seeks to facilitate detailed observation of lung tissue movement and changes. The effectiveness of image registration is evaluated using the Target Registration Error (TRE).

## Objectives
1. Develop an algorithm for registering MRI lung images between exhalation and inhalation phases.
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
The project successfully achieved automated registration of lung MRI images, with accuracy validated by TRE calculations. Detailed results are presented in the Jupyter notebooks.

## Authors
- [Colin Tenorio](https://github.com/CarmenColinTen)
- [Edwing Ulin](https://github.com/EdAlita)

## License
This project is licensed under the Creative Common Lincense - see the [LICENSE.md](LICENSE) file for details.

## Acknowledgments
- To my mental stability for not pulling the trigger
- ChatGPT