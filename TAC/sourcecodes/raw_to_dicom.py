import numpy as np
import pydicom
from pydicom.dataset import FileDataset
import datetime
import os

# ------------------ DICOM Slice Creation ------------------

def create_dicom_slice(
    slice_array,
    filename,
    slice_index,
    study_uid,
    series_uid,
    patient_id
):
    file_meta = pydicom.Dataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID

    ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)

    now = datetime.datetime.now()
    ds.ContentDate = now.strftime("%Y%m%d")
    ds.ContentTime = now.strftime("%H%M%S")

    ds.PatientName = "Anonymous"
    ds.PatientID = patient_id

    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID

    ds.Modality = "CT"
    ds.SeriesNumber = 1
    ds.InstanceNumber = slice_index + 1

    ds.ImagePositionPatient = [0.0, 0.0, float(slice_index)]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.PixelSpacing = [1.0, 1.0]
    ds.SliceThickness = 1.0

    ds.Rows, ds.Columns = slice_array.shape
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"

    ds.PixelData = slice_array.tobytes()
    ds.save_as(filename)

# ------------------ RAW → DICOM Conversion ------------------

def raw_to_dicom(
    raw_path,
    output_dir,
    width=512,
    height=512,
    dtype=np.uint8
):
    os.makedirs(output_dir, exist_ok=True)

    data = np.fromfile(raw_path, dtype=dtype)

    pixels_per_slice = width * height
    nslices = data.size // pixels_per_slice
    remainder = data.size % pixels_per_slice

    if nslices == 0:
        raise ValueError("RAW file too small to contain valid CT slices.")

    if remainder != 0:
        print(f"[WARNING] {raw_path}: ignoring {remainder} extra bytes")

    volume = data[:nslices * pixels_per_slice].reshape(
        (height, width, nslices)
    )

    study_uid = pydicom.uid.generate_uid()
    series_uid = pydicom.uid.generate_uid()

    patient_id = os.path.basename(raw_path)

    for i in range(nslices):
        filename = os.path.join(output_dir, f"slice_{i+1:04d}.dcm")
        create_dicom_slice(
            volume[:, :, i],
            filename,
            i,
            study_uid,
            series_uid,
            patient_id
        )

    print(f"[OK] {patient_id} → {nslices} slices converted")

# ------------------ Batch Processing ------------------

if __name__ == "__main__":

    raw_base_dir = "."              # folder containing TC_0001 ... TC_0006
    output_base_dir = "DICOM"

    for i in range(1, 7):
        patient_id = f"TC_000{i}"
        raw_file = os.path.join(raw_base_dir, patient_id)
        output_dir = os.path.join(output_base_dir, patient_id)

        if not os.path.exists(raw_file):
            print(f"[SKIP] {patient_id} not found")
            continue

        try:
            raw_to_dicom(
                raw_path=raw_file,
                output_dir=output_dir,
                dtype=np.uint8
            )
        except Exception as e:
            print(f"[ERROR] {patient_id}: {e}")
