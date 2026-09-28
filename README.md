# DICOM_Preprocessing_Uisg_MONAI


Below is a cleaned-up, GitHub-ready version. I have incorporated the `KeyError: 'pixdim'` issue and corrected the voxel-spacing section so that it uses MONAI's affine matrix rather than assuming a `pixdim` metadata field.

# Chest CT Preprocessing with MONAI

A practical introduction to preprocessing a 3D chest CT directly from a DICOM series using [MONAI](https://monai.io/) and PyTorch.

This tutorial demonstrates a basic medical-imaging preprocessing pipeline without converting the CT data to NIfTI:

```text
DICOM series
     ↓
Load with MONAI
     ↓
3D CT volume
     ↓
Orientation standardization
     ↓
Voxel-spacing standardization
     ↓
Intensity normalization
     ↓
Data augmentation
     ↓
PyTorch tensor
```

The purpose of this tutorial is to understand what each preprocessing step does, why it is needed, and how it changes the CT volume.

---

## 1. Prerequisites

The examples use:

* Python 3.9+
* PyTorch
* MONAI
* pydicom
* NumPy
* Matplotlib

Install the required packages:

```python
!pip install monai torch pydicom numpy matplotlib
```

After installation, restart the Jupyter kernel if necessary.

---

# 2. Import the Required Libraries

```python
import os

import numpy as np
import matplotlib.pyplot as plt
import torch

from monai.data import PydicomReader

from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    RandFlipd,
    RandRotate90d,
)
```

---

# 3. Organize the DICOM Data

This tutorial assumes that the CT scan is stored as a **DICOM series**, meaning that one patient scan consists of multiple DICOM files representing individual slices.
You can download an exmaple CT from "https://saga-it.com/dicom/samples/files/ct-chest-lidc-idri/series.zip"
A typical directory may look like:

```text
project/
│
├── data/
│   └── dicom/
│       ├── slice001.dcm
│       ├── slice002.dcm
│       ├── slice003.dcm
│       ├── ...
│       └── slice250.dcm
│
└── notebook.ipynb
```

The exact directory structure is not important as long as the DICOM series can be identified.

---

# 4. Locate the DICOM Series

The following code searches for a directory containing DICOM files.

```python
dicom_root = "data/dicom"

dicom_series_folder = None

for root, folders, files in os.walk(dicom_root):

    dcm_files = [
        f for f in files
        if f.lower().endswith(".dcm")
    ]

    if len(dcm_files) > 10:
        dicom_series_folder = root
        break

if dicom_series_folder is None:
    raise FileNotFoundError(
        "No DICOM series containing more than 10 .dcm files was found."
    )

print("DICOM series:")
print(dicom_series_folder)
```

This is preferable to assuming a particular folder name because different datasets organize DICOM files differently.

---

# 5. Load the DICOM Series with MONAI

MONAI provides `PydicomReader`, which can be used to read a DICOM series directly.

```python
data = {
    "image": dicom_series_folder
}

loader = LoadImaged(
    keys=["image"],
    reader=PydicomReader,
    image_only=False
)

data = loader(data)
```

At this point, MONAI has reconstructed the individual DICOM slices into a 3D image volume.

Add the channel dimension:

```python
channel_first = EnsureChannelFirstd(
    keys=["image"]
)

data = channel_first(data)
```

Check the resulting shape:

```python
print("CT shape:")
print(data["image"].shape)
```

A typical result may look like:

```text
torch.Size([1, 512, 512, 250])
```

The dimensions represent:

```text
[Channel, Height, Width, Depth]
```

The exact dimensions depend on the CT acquisition.

---

# 6. Visualize the Original CT

A CT volume contains many axial slices. We can display the middle slice as an initial check.

```python
ct = data["image"]

middle_slice = ct.shape[-1] // 2

plt.figure(figsize=(6, 6))

plt.imshow(
    ct[0, :, :, middle_slice],
    cmap="gray",
    vmin=-1000,
    vmax=400
)

plt.title("Original DICOM CT")
plt.axis("off")

plt.show()
```

The intensity window from approximately `-1000` to `400 HU` is commonly useful for visualizing lung CT data.

This visualization is only for display. The underlying CT values have not yet been normalized.

---

# 7. Inspect the Spatial Metadata

Spatial information is extremely important in medical imaging.

A CT image is not simply a 3D array of numbers. The voxel values are associated with a physical coordinate system through metadata and an affine transformation.

A common mistake is to assume that MONAI will always expose voxel spacing through a metadata field called `pixdim`.

For example, the following code may fail:

```python
print(data["image"].meta["pixdim"])
```

with:

```text
KeyError: 'pixdim'
```

## Why does this happen?

MONAI does not guarantee that the metadata dictionary will contain a key named `pixdim`.

Instead, the `MetaTensor` provides an **affine matrix**, which describes the relationship between voxel coordinates and physical coordinates.

Use the affine matrix to obtain voxel spacing.

```python
# Get the 4 x 4 affine matrix
affine = data["image"].affine

# The lengths of the first three columns correspond
# to the physical size of the voxel along each axis.
spacing = np.sqrt(
    (affine[:3, :3] ** 2).sum(axis=0)
)

print("Original voxel spacing (mm):")
print(spacing)
```

For example, you might obtain:

```text
Original voxel spacing (mm):
[0.70 0.70 1.00]
```

This means that the reconstructed voxel grid has approximately:

```text
0.70 mm × 0.70 mm × 1.00 mm
```

spacing along its three axes.

You can also inspect the metadata available in the `MetaTensor`:

```python
print(data["image"].meta.keys())
```

### Important

Do not assume that a specific metadata key such as `pixdim` exists. For spatial calculations, using the affine is more robust.

---

# 8. Why Voxel Spacing Matters

Two CT scans can contain the same anatomical structures but use different voxel sizes.

For example:

```text
CT A:
0.7 × 0.7 × 1.0 mm

CT B:
0.9 × 0.9 × 2.5 mm
```

The numerical array dimensions are therefore not directly comparable.

A voxel represents a physical volume in the patient.

If a deep-learning model receives CT scans acquired at substantially different spatial resolutions, the same anatomical structure can occupy very different numbers of voxels.

Therefore, many medical-imaging pipelines standardize the voxel spacing before training.

---

# 9. Standardize the Image Orientation

Different imaging systems can store anatomical axes in different orientations.

MONAI can reorient the volume to a consistent coordinate convention.

Here we use RAS:

```python
before_orientation = data["image"].clone()

orientation = Orientationd(
    keys=["image"],
    axcodes="RAS"
)

data = orientation(data)

after_orientation = data["image"]
```

The meaning of RAS is:

```text
R = Right
A = Anterior
S = Superior
```

This does not necessarily change the physical anatomy. It establishes a consistent ordering of the image axes.

Visual comparison:

```python
before_slice = before_orientation.shape[-1] // 2
after_slice = after_orientation.shape[-1] // 2

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.imshow(
    before_orientation[0, :, :, before_slice],
    cmap="gray",
    vmin=-1000,
    vmax=400
)

plt.title("Before orientation")
plt.axis("off")


plt.subplot(1, 2, 2)

plt.imshow(
    after_orientation[0, :, :, after_slice],
    cmap="gray",
    vmin=-1000,
    vmax=400
)

plt.title("After orientation: RAS")
plt.axis("off")

plt.show()
```

The images may look almost identical. That is expected because orientation standardization primarily concerns the spatial coordinate system.

---

# 10. Resample the CT to a Standard Voxel Spacing

Suppose different CT scans have different voxel spacings.

We can resample them onto a common grid.

For this example:

```text
1.0 × 1.0 × 1.0 mm
```

Use MONAI's `Spacingd` transform:

```python
before_spacing = data["image"].clone()

spacing = Spacingd(
    keys=["image"],
    pixdim=(1.0, 1.0, 1.0),
    mode="bilinear"
)

data = spacing(data)

after_spacing = data["image"]

print("Before resampling:", before_spacing.shape)
print("After resampling: ", after_spacing.shape)
```

The spatial dimensions may change after resampling.

For example:

```text
Before:
[1, 512, 512, 250]

After:
[1, 560, 560, 310]
```

The exact values depend on the original image dimensions and voxel spacing.

---

# 11. Visualize the Resampled CT

```python
before_slice = before_spacing.shape[-1] // 2
after_slice = after_spacing.shape[-1] // 2

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.imshow(
    before_spacing[0, :, :, before_slice],
    cmap="gray",
    vmin=-1000,
    vmax=400
)

plt.title("Before resampling")
plt.axis("off")


plt.subplot(1, 2, 2)

plt.imshow(
    after_spacing[0, :, :, after_slice],
    cmap="gray",
    vmin=-1000,
    vmax=400
)

plt.title("After resampling: 1 mm isotropic")
plt.axis("off")

plt.show()
```

Resampling does not create new anatomical information. It interpolates the image onto a different physical grid.

For CT images, linear interpolation is commonly appropriate because intensity values vary continuously.

---

# 12. Normalize CT Intensity

CT intensity is expressed in **Hounsfield Units (HU)**.

Typical values include approximately:

```text
Air       ≈ -1000 HU
Water     ≈ 0 HU
Soft tissue > 0 HU
Dense bone ≫ 0 HU
```

For lung-focused applications, it is often useful to restrict the relevant intensity range.

Here we clip the CT to:

```text
[-1000, 400] HU
```

and map the result to:

```text
[0, 1]
```

```python
before_normalization = data["image"].clone()

normalization = ScaleIntensityRanged(
    keys=["image"],
    a_min=-1000,
    a_max=400,
    b_min=0.0,
    b_max=1.0,
    clip=True
)

data = normalization(data)

after_normalization = data["image"]
```

The transformation is conceptually:

$$
x_{\text{normalized}}
=
\frac{\operatorname{clip}(x,-1000,400)+1000}
{1400}
$$

Therefore:

```text
-1000 HU → 0.0
   400 HU → 1.0
```

Values outside the range are clipped.

---

# 13. Visualize Intensity Normalization

```python
before_slice = before_normalization.shape[-1] // 2
after_slice = after_normalization.shape[-1] // 2

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.imshow(
    before_normalization[0, :, :, before_slice],
    cmap="gray",
    vmin=-1000,
    vmax=400
)

plt.title("Before normalization")
plt.axis("off")


plt.subplot(1, 2, 2)

plt.imshow(
    after_normalization[0, :, :, after_slice],
    cmap="gray",
    vmin=0,
    vmax=1
)

plt.title("After normalization")
plt.axis("off")

plt.show()
```

The visual appearance may remain very similar because both images are displayed using equivalent intensity ranges.

The important difference is the numerical representation.

---

# 14. Data Augmentation

Preprocessing makes the input consistent.

Augmentation is different. It deliberately modifies training images so that a model learns features that are less sensitive to small spatial variations.

For example, we can randomly flip the volume.

```python
before_augmentation = data["image"].clone()

flip = RandFlipd(
    keys=["image"],
    prob=0.5,
    spatial_axis=0
)

data = flip(data)

after_augmentation = data["image"]
```

The probability is `0.5`, meaning that the transformation is applied approximately half the time.

For demonstration purposes, you can force the transformation by changing:

```python
prob=0.5
```

to:

```python
prob=1.0
```

---

# 15. Rotation Augmentation

MONAI also provides 90-degree rotation augmentation.

```python
before_rotation = data["image"].clone()

rotation = RandRotate90d(
    keys=["image"],
    prob=0.5,
    max_k=1,
    spatial_axes=(0, 1)
)

data = rotation(data)

after_rotation = data["image"]
```

Here:

* `prob=0.5` means the rotation is applied with probability 0.5.
* `max_k=1` restricts the transformation to at most one 90-degree rotation.
* `spatial_axes=(0, 1)` specifies the two spatial axes involved in the rotation.

Augmentations should be chosen according to the medical problem. A transformation that is reasonable for one imaging task may be inappropriate for another.

---

# 16. Convert the Result to a PyTorch Tensor

MONAI can represent images as `MetaTensor` objects, which contain both tensor data and spatial metadata.

For a deep-learning model, the numerical tensor can be accessed directly.

```python
ct_tensor = torch.as_tensor(
    data["image"],
    dtype=torch.float32
)

print("Final tensor shape:")
print(ct_tensor.shape)
```

A typical result may look like:

```text
torch.Size([1, Z, Y, X])
```

For example:

```text
torch.Size([1, 310, 560, 560])
```

The first dimension represents the single CT channel.

---

# 17. Complete Preprocessing Pipeline

The complete process can be summarized as:

```text
DICOM files
    │
    ▼
MONAI PydicomReader
    │
    ▼
3D CT volume
    │
    ▼
Ensure channel-first format
    │
    ▼
Orientation standardization
    │
    ▼
Voxel-spacing standardization
    │
    ▼
HU clipping and normalization
    │
    ▼
Training augmentation
    │
    ▼
PyTorch tensor
```

In code:

```python
data = {
    "image": dicom_series_folder
}

loader = LoadImaged(
    keys=["image"],
    reader=PydicomReader,
    image_only=False
)

data = loader(data)

data = EnsureChannelFirstd(
    keys=["image"]
)(data)

data = Orientationd(
    keys=["image"],
    axcodes="RAS"
)(data)

data = Spacingd(
    keys=["image"],
    pixdim=(1.0, 1.0, 1.0),
    mode="bilinear"
)(data)

data = ScaleIntensityRanged(
    keys=["image"],
    a_min=-1000,
    a_max=400,
    b_min=0.0,
    b_max=1.0,
    clip=True
)(data)
```

Augmentation can then be applied during model training:

```python
data = RandFlipd(
    keys=["image"],
    prob=0.5,
    spatial_axis=0
)(data)
```

---

# 18. Important Distinction: Preprocessing vs Augmentation

These two concepts should not be treated as the same operation.

### Preprocessing

The purpose is to make the input consistent.

Examples:

```text
Orientation
Voxel spacing
Intensity normalization
```

These transformations are generally part of the standard input pipeline.

### Augmentation

The purpose is to create controlled variations of training data.

Examples:

```text
Random flipping
Random rotation
Random cropping
Random intensity transformations
```

Augmentation is normally applied during model training rather than permanently modifying the dataset.

---

# 19. Why We Do Not Convert the CT to NIfTI

This tutorial deliberately works directly with the DICOM series.

The pipeline is:

```text
DICOM → MONAI → PyTorch
```

There is no requirement to perform:

```text
DICOM → NIfTI → MONAI
```

DICOM is the original clinical imaging format and contains important acquisition and spatial metadata. MONAI can read the DICOM series directly through its DICOM reader.

For a learning exercise, working directly with the DICOM series is useful because it makes the relationship between the original clinical data, spatial metadata, voxel spacing, and the resulting 3D tensor easier to understand.

---

# 20. Troubleshooting

## `KeyError: 'pixdim'`

Error:

```text
KeyError: 'pixdim'
```

Problematic code:

```python
data["image"].meta["pixdim"]
```

Use the affine instead:

```python
affine = data["image"].affine

spacing = np.sqrt(
    (affine[:3, :3] ** 2).sum(axis=0)
)

print("Original voxel spacing (mm):")
print(spacing)
```

You can inspect the available metadata with:

```python
print(data["image"].meta.keys())
```

The important lesson is not to assume that every MONAI image will expose spatial information under exactly the same metadata key.

---

# 21. Conceptual Takeaway

A CT scan is not simply a stack of 2D images.

It is a 3D volume in which every voxel has:

```text
1. An intensity value
2. A physical position
3. A physical size
```

For example:

```text
Intensity:
    -850 HU

Voxel spacing:
    0.7 × 0.7 × 1.0 mm

Position:
    determined through the image's spatial metadata/affine
```

Therefore, a medically meaningful preprocessing pipeline must consider both the **voxel values** and the **physical geometry** of the image.

The MONAI workflow in this tutorial addresses these components explicitly:

```text
DICOM
  ↓
Read the 3D volume
  ↓
Understand spatial metadata
  ↓
Standardize orientation
  ↓
Standardize voxel spacing
  ↓
Normalize CT intensity
  ↓
Apply training augmentation
  ↓
Feed the resulting tensor to a deep-learning model
```

This provides the basic foundation required before moving to more advanced tasks such as lung segmentation, 3D CNNs, Vision Transformers, CT-ViT, or multimodal CT-text models.

This version is suitable as a `README.md` or as the explanatory markdown accompanying a Jupyter notebook. The most important correction is that voxel spacing is derived from `data["image"].affine`, rather than assuming `data["image"].meta["pixdim"]` exists.
