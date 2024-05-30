from pathlib import Path
from tqdm.auto import tqdm
import json
import numpy as np
import scipy.io
import SimpleITK as sitk
from resample_image import resample_sitk_image
import argparse

"""
Example usage:
python3 prepare_farsiu.py --input_folder=/path/to/original_farsiu_folder/ --nifti_folder=/path/to/nifti/raw --resampled_folder=/path/to/nifti/resampled
"""

spacing = [0.013899999670684338, 0.0038999998942017555, -1.0]


def resample(nifti_folder, resampled_folder, spacing):
    paths = list(nifti_folder.glob("*.nii.gz"))

    for path in tqdm(paths):
        im = sitk.ReadImage(str(path))
        old_im_spacing = im.GetSpacing()
        new_im_spacing = [
            s if s != -1 else old_im_spacing[idx] for idx, s in enumerate(spacing)
        ]

        new_im = resample_sitk_image(im, new_im_spacing, interpolator="linear")
        filter = sitk.MinimumMaximumImageFilter()
        filter.Execute(new_im)
        new_im = (
            (new_im - filter.GetMinimum())
            / (filter.GetMaximum() - filter.GetMinimum())
            * 255
        )
        new_im = sitk.Cast(new_im, sitk.sitkUInt8)

        sitk.WriteImage(new_im, str(resampled_folder / path.name), True)


def make_labels(folder):
    paths = folder.glob("*.nii.gz")
    labels = {}
    for path in paths:
        name = path.name[: -len(".nii.gz")]
        labels[name] = int("AMD" in name)
    print(labels)
    with open(folder / "labels.json", "w") as f:
        f.write(json.dumps(labels, indent=4))


def mat_to_nifti(input_folder, nifti_folder, subset):
    files = (input_folder / subset).glob("*.mat")
    for file in files:
        nr = file.stem.split("_")[-1]
        out_file = nifti_folder / f"{subset}{nr}.nii.gz"

        # Swapaxes and rot90 are necessary to put the volume in the right
        # orientation for SimpleITK.
        im = scipy.io.loadmat(file)["images"].swapaxes(0, 2)
        im = np.rot90(im, 3, (1, 2))
        sim = sitk.GetImageFromArray(im)

        in_spacing_mm = np.array((6.7, 2, 6.7)) / sim.GetSize()

        sim.SetSpacing(in_spacing_mm)
        sitk.WriteImage(sim, str(out_file))
        print("Wrote", out_file)


def make_empty_labels(folder):
    with open(folder / "labels.json", "w") as f:
        f.write(json.dumps({}, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", default="Farsiu")
    parser.add_argument("--nifti_folder", default="Farsiu_NIFTI/raw")
    parser.add_argument("--resampled_folder", default="Farsiu_NIFTI/resampled")
    args = parser.parse_args()
    
    input_folder = Path(args.input_folder)
    nifti_folder = Path(args.nifti_folder)
    resampled_folder = Path(args.resampled_folder)

    for subset in ["train", "val", "test"]:
        (nifti_folder / subset).mkdir(exist_ok=True, parents=True)
        (resampled_folder / subset).mkdir(exist_ok=True, parents=True)

    nifti_folder = nifti_folder / "test"
    resampled_folder = resampled_folder / "test"

    nifti_folder.mkdir(exist_ok=True, parents=True)
    resampled_folder.mkdir(exist_ok=True, parents=True)

    mat_to_nifti(input_folder, nifti_folder, "AMD")
    mat_to_nifti(input_folder, nifti_folder, "Control")
    resample(nifti_folder, resampled_folder, spacing)
    make_labels(nifti_folder)
    make_labels(resampled_folder)

    make_empty_labels(nifti_folder / "train")
    make_empty_labels(nifti_folder / "val")
