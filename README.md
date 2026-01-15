**Data Privacy**

This repository contains private and sensitive medical data and is provided strictly for research and educational purposes. 
Users must handle the data with care and comply with all applicable ethical guidelines, data protection regulations, and institutional review requirements.
Redistribution, re-identification attempts, or use for commercial or clinical decision-making is strictly prohibited. 
By using this dataset, you agree to take full responsibility for ensuring appropriate data security and lawful use.
If you use this dataset in any publication, presentation, or derived work, please give proper credit to the original data source as specified by the authors.

_______________________________________________________________________________________________________________________________________


# Alzheimer Disease Decoding using EEG Data

## 1. Dataset Overview

EEG dataset used for the classification/decoding of Alzheimer (AD), Creutzfeldt Jacob-disease (CJD), and Healthy Control (CNTRL) subjects.
For the dataset, look into Release 1 or use this link: https://github.com/AI-Lab-UniRC/FAIR-NAEL-Database/releases/tag/EEG_AD_NAEL
<<<<<<< HEAD

The zip folder contains three subfolders named AD, CJD, CNTRL.

Each subbfolder includes further two sub-folders named as X_TXT and X_XLSX.

X_TXT is a folder with raw data files of all subjects related to X disease/class.

X_XLSX is a folder that contains the information of signal discontinuity from start to end (da:start and a:end) over the time, for all subjects related to X disease/class.

=======

* The zip folder contains three subfolders named **AD**, **CJD**, **CNTRL**.

* Each subbfolder includes further two sub-folders named as **X_TXT** and **X_XLSX**.

* **X_TXT** is a folder with raw data files of all subjects related to X disease/class.

* **X_XLSX** is a folder that contains the information of signal discontinuity from start to end (da:start and a:end) over the time, for all subjects related to X disease/class.

>>>>>>> bf16613 (both datasets update)
The dataset is balanced with each class having 12 subjects.


## 2. Dataset Size
| Classes | No. of Subjects |
| :--- | :--- | 
| **AD** | 12 | 
| **CJD** | 12 | 
| **Control** | 12 |


## 3. EEG Data Processign and Decoding

#### The python file *fair_nael_source codes.py* contains the complete codes to reproduce the results for decoding EEG data for AD, CJD, Contrl subjects.
* You have to download the repository and install the torch (main requirement).
<<<<<<< HEAD
- Configure hyperparametrs (kfolds, batch_size, etc) as per your requirements, from the botoom of the .py file.
- Process the EEG data, since original row data is provided for processing and cleaning of the data as per your requirements or cleaning approach.
- For running the proposed model, for LOSO evaluation, you have to provide the subject_ids, data, and labels to the training function.
- By this it will run on your available GPU/COU and compute the rsults and save them in a csv file on your disk.
=======
* Configure hyperparametrs (kfolds, batch_size, etc) as per your requirements, from the botoom of the .py file.
* Process the EEG data, since original row data is provided for processing and cleaning of the data as per your requirements or cleaning approach.
* For running the proposed model, for LOSO evaluation, you have to provide the subject_ids, data, and labels to the training function.
* By this it will run on your available GPU/COU and compute the rsults and save them in a csv file on your disk.

>>>>>>> bf16613 (both datasets update)
Do not hesitate to get in touch for any details or having issues in running the experiments.

_______________________________________________________________________________________________________________________________________


# Recto-Colon Cancer TAC Dataset Report

## 1. Dataset Overview
A medical imaging dataset composed of six files with no file extension. Each file corresponds to a single patient CT scan (TAC - Tomografía Axial Computarizada), representing recto-colon cancer cases. The files are stored in a raw binary format, where each file represents a complete 3D volumetric scan of the patient.
For the dataset, look into Release 1 or use this link: https://github.com/AI-Lab-UniRC/FAIR-NAEL-Database/releases/tag/TAC_NAEL_FAIR

## 2. Technical Specifications
* **File Format:** Raw binary data
* **Data Type:** 8-bit unsigned integers (`uint8`)
* **Spatial Resolution:** 512 × 512 pixels per slice 


## 3. Volume Specifications
The dataset contains volumetric data with the following slice counts and file sizes:

| Patient | Dimensions | Slices | Size (MB) |
| :--- | :--- | :--- | :--- |
| **TC_0001** | (512, 512) | 2096 | 524 |
| **TC_0002** | (512, 512) | 2408 | 602 |
| **TC_0003** | (512, 512) | 1816 | 454 |
| **TC_0004** | (512, 512) | 1896 | 474 |
| **TC_0005** | (512, 512) | 2664 | 666 |
| **TC_0006** | (512, 512) | 2216 | 554 |

## 4. Data Conversion & Usage
To facilitate analysis with standard medical imaging software, custom Python scripts (using `nibabel`, `pydicom`, etc.) were used to convert the raw data.

### Supported Formats
1.  **NIfTI (`.nii`)**
    * Designed for use with **3D Slicer**.
    * Can be loaded as a **Volume** (standard image scrolling) or as a **Segmentation** (label map).
2.  **DICOM (`.dcm`)**
    * Designed for use with **RadiAnt DICOM Viewer**.
    * Allows for 3D reconstruction, rotation, zooming, and real-time slicing.
________________________________________________________________________________________________________________________________________

#### Acknowledgements

TBA.
