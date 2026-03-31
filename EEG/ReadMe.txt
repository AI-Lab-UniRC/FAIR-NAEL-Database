## EEG Data processing, Decoding, Model Testing, and Edge-AI Deployment

### Overview
This folder 'sourcecodes' contains a python file 'fail_nael_source_codes.py' which focuses on training and deploying od subject-specific decoding models for real-time inference on Edge-AI devices. Python file *fair_nael_source codes.py* contains the complete codes to reproduce the results for decoding EEG data for AD, CJD, Contrl subjects.

#### Pipeline
- Train and test models locally using GPU
- Export trained models to **TorchScript**
- Deploy and validate on **NVIDIA Jetson AGX**

---

### System Setup

#### Training Environment
- **GPU**: NVIDIA RTX 4000 (CUDA-enabled)
- **Framework**: PyTorch
- **Export Format**: TorchScript (`.pt`)

#### Edge-AI Environment
- **Device**: NVIDIA Jetson AGX
- **Runtime**: PyTorch (TorchScript inference)

---

### Model Training & Testing

- Models are trained **per subject**
- Each subject has:
  - A dedicated trained model
  - A corresponding test dataset

- After training:
  - Models are evaluated locally on GPU
  - Performance is validated before deployment

---

### Model Export (TorchScript)

TorchScript is used to serialize models for deployment and remove Python dependency during inference.

Do not hesitate to get in touch for any details or having issues in running the experiments.
