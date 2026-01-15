import numpy as np
import nibabel as nib
import os

def raw_to_nifti(
    raw_path,
    output_nii,
    width=512,
    height=512,
    dtype=np.uint8,
    voxel_spacing=(1.0, 1.0, 1.0)
):
    data = np.fromfile(raw_path, dtype=dtype)

    pixels_per_slice = width * height
    nslices = data.size // pixels_per_slice
    remainder = data.size % pixels_per_slice

    if nslices == 0:
        raise ValueError("RAW file too small to contain valid slices.")

    if remainder != 0:
        print(f"[WARNING] {raw_path}: ignoring {remainder} extra bytes")

    volume = data[:nslices * pixels_per_slice].reshape(
        (height, width, nslices)
    )

    affine = np.eye(4)
    affine[0, 0] = voxel_spacing[0]
    affine[1, 1] = voxel_spacing[1]
    affine[2, 2] = voxel_spacing[2]

    nii = nib.Nifti1Image(volume, affine)
    nib.save(nii, output_nii)

    print(f"[OK] Saved NIfTI: {output_nii}")
    print(f"[INFO] Detected slices: {nslices}")


# ------------------ Batch Processing ------------------

if __name__ == "__main__":

    raw_base_dir = "."          # folder containing TC_0001 ... TC_0006
    output_base_dir = "NIFTI"

    os.makedirs(output_base_dir, exist_ok=True)

    for i in range(1, 7):
        patient_id = f"TC_000{i}"
        raw_file = os.path.join(raw_base_dir, patient_id)
        output_nii = os.path.join(output_base_dir, f"{patient_id}.nii")

        if not os.path.exists(raw_file):
            print(f"[SKIP] {patient_id} not found")
            continue

        try:
            raw_to_nifti(
                raw_path=raw_file,
                output_nii=output_nii,
                dtype=np.uint8
            )
        except Exception as e:
            print(f"[ERROR] {patient_id}: {e}")
