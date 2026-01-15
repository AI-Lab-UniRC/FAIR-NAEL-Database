RAW CT Data Conversion

This repository provides Python scripts to convert anonymized 3D CT scans stored as RAW files (512×512×N, uint8) into standard medical imaging formats.

RAW → DICOM
Converts each RAW volume into a DICOM slice series compatible with RadiAnt DICOM Viewer, supporting 3D reconstruction and interactive slicing.

RAW → NIfTI
Converts each RAW volume into a NIfTI (.nii) file compatible with 3D Slicer for visualization and segmentation.

Both scripts automatically infer the number of slices from the file size.
