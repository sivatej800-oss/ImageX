# Brain MRI Segmentation Project - Pipeline Documentation

This project performs **brain tumor segmentation** on MRI scans using two distinct approaches:

1. **nnU-Net Pipeline** - Uses a pre-trained nnU-Net model for multi-class tumor segmentation (inference only)
2. **SwinUNETR Pipeline** - A full preprocessing-to-training pipeline using a self-supervised pre-trained Swin UNETR transformer for binary tumor segmentation

Both pipelines operate on **BraTS (Brain Tumor Segmentation)** challenge data, which provides multi-modal 3D MRI volumes with expert-annotated tumor masks.

---

## The Data: BraTS MRI Scans

Each patient case in the BraTS dataset consists of four 3D MRI modalities (each a separate `.nii` file):

| Channel | Modality | What it shows |
|---------|----------|---------------|
| 0 | **T1** (T1-weighted) | Basic brain anatomy |
| 1 | **T1ce** (T1 contrast-enhanced) | Highlights active/enhancing tumor regions |
| 2 | **T2** (T2-weighted) | Shows edema (swelling) around tumors |
| 3 | **FLAIR** (Fluid-Attenuated Inversion Recovery) | Highlights lesions near fluid-filled areas |

Plus a **segmentation mask** (`seg.nii`) with these labels:
- `0` = Background (healthy tissue)
- `1` = Peritumoral edema (swelling)
- `2` = Non-enhancing / necrotic tumor core
- `4` = Enhancing tumor (active, gadolinium-enhancing)

Two sample cases are included in this project:
- `BraTS2021_00495` (BraTS 2021 glioma)
- `BraTS-MET-00086-000` (BraTS brain metastasis)

---

## Pipeline 1: nnU-Net (Inference / Evaluation)

**Notebooks:** `Brain_MRI_segmentation.ipynb` and `Brain_MRI_segmentation_2.ipynb`

This pipeline uses a **pre-trained nnU-Net v2** model to segment brain tumors. nnU-Net is a "self-configuring" framework that automatically adapts its architecture and training strategy to any biomedical segmentation dataset. Here, the model was already trained on the BraTS 2019 dataset (80,064 training samples) using 5-fold cross-validation, so this pipeline only performs **inference** (prediction) on new cases.

### Step-by-step:

#### Step 1: Environment Setup
```
pip install nnunetv2 nibabel medpy SimpleITK
```
Runs on Google Colab with Google Drive mounted. Sets the `nnUNet_results` environment variable to point to the folder containing the trained model checkpoints.

#### Step 2: Prepare Input Files in nnU-Net Format
nnU-Net expects a very specific naming convention. Each MRI modality must be saved as a separate `.nii.gz` file with a channel suffix:

```
BraTS2021_00495_0000.nii.gz  ->  T1       (channel 0)
BraTS2021_00495_0001.nii.gz  ->  T1ce     (channel 1)
BraTS2021_00495_0002.nii.gz  ->  T2       (channel 2)
BraTS2021_00495_0003.nii.gz  ->  FLAIR    (channel 3)
```

The raw `.nii` files are loaded with `nibabel` and re-saved as compressed `.nii.gz` into an `imagesTs/` folder.

#### Step 3: Run nnU-Net Prediction
```bash
nnUNetv2_predict \
  -i /content/imagesTs \       # input images folder
  -o /content/preds \          # output predictions folder
  -d Dataset002_BRATS19 \      # dataset name (matches trained model)
  -c 3d_fullres \              # configuration: full-resolution 3D
  -tr nnUNetTrainer \          # trainer class
  -p nnUNetPlans \             # planning configuration
  -f 0                         # fold number (0 = single fold; use "0 1 2 3 4" for ensemble)
```

**What happens under the hood:**
- The model loads its architecture from `plans.json`: a 3D PlainConvUNet with 6 encoder stages, 5 decoder stages, patch size 128x160x112, batch size 2
- All 4 input channels are Z-score normalized using brain-mask-aware statistics
- The image is processed at 1mm isotropic spacing
- Prediction is done with a sliding window approach and test-time augmentation (TTA)
- Output is a 3D segmentation mask with labels 0, 1, 2, 4

#### Step 4: Evaluate with Dice Score
The predicted mask is compared against the ground truth using the **Dice coefficient** (a standard overlap metric where 1.0 = perfect match):

```python
from medpy.metric.binary import dc
# Whole-tumor: is any tumor voxel correct?
dc(pred > 0, gt > 0)
# Per-class: how well does each tumor sub-region match?
{1: dc(pred==1, gt==1),  2: dc(pred==2, gt==2),  4: dc(pred==4, gt==4)}
```

**Results on BraTS2021_00495 (glioma):**
| Metric | Score |
|--------|-------|
| Whole-tumor Dice | 0.966 |
| Edema (label 1) | 0.974 |
| Necrotic core (label 2) | 0.915 |
| Enhancing tumor (label 4) | 0.948 |

**Results on BraTS-MET-00086-000 (metastasis):**
| Metric | Score |
|--------|-------|
| Whole-tumor Dice | 0.493 |
| Edema (label 1) | 0.000 |
| Necrotic core (label 2) | 0.473 |
| Enhancing tumor (label 4) | 0.000 |

The poor performance on the metastasis case is expected -- the model was trained on gliomas (BraTS 2019), which have very different tumor morphology than metastases.

#### Step 5: Visualize Results
A matplotlib figure shows three panels at a representative axial slice:
1. Ground truth segmentation (color-coded by label)
2. Model prediction (same color map)
3. FLAIR image with contour overlays (white = GT boundary, yellow = prediction boundary)

### nnU-Net Model Details
- **Architecture:** PlainConvUNet (3D), 6 encoder / 5 decoder stages, max 320 features
- **Training data:** 80,064 samples from BraTS 2019 (Dataset002_BRATS19, originally from the BAMF dataset)
- **Cross-validation:** 5 folds, each with a `checkpoint_final.pth`
- **Input:** 4-channel 3D volumes at 1mm isotropic spacing
- **Output:** 5-class segmentation (background + 4 tumor sub-regions, though label 3 is unused/"empty")

---

## Pipeline 2: SwinUNETR (Preprocessing + Fine-tuning for Segmentation)

**Notebooks:** `Final_PreProcess_SwinBrain.ipynb` (data prep) and `Fine_tune_commented.ipynb` (training)
**Source code:** `SwinBrain/` directory

This pipeline takes raw BraTS MRI data through a full preprocessing workflow, then fine-tunes a **self-supervised pre-trained SwinUNETR** model for binary tumor segmentation. Unlike the nnU-Net approach, this pipeline collapses all tumor sub-classes into a single "tumor" class.

### Preprocessing Steps (Final_PreProcess_SwinBrain.ipynb)

#### Step 1: Skull Stripping (T1)
Extracts just the brain tissue from the T1 scan, removing the skull.

- **Option A (simple):** Since BraTS data is already skull-stripped, create a binary brain mask by thresholding (voxels > 0 = brain).
- **Option B (HD-BET):** Run the HD-BET deep learning tool for proper skull stripping on non-preprocessed data.

**Outputs:** `t1_brain.nii.gz` + `t1_brain_mask.nii.gz`

#### Step 2: Crop to Brain Region (T1)
Computes a tight 3D bounding box around the non-zero voxels of the brain mask, then crops the T1 image and mask to remove empty background space. The NIfTI affine matrix is adjusted so spatial coordinates remain correct.

**Output:** `t1_brain_cropped.nii.gz`, `t1_brain_mask_cropped.nii.gz`

#### Step 3: Co-register T2w and FLAIR to T1
The T2-weighted and T2-FLAIR scans may be in slightly different spatial positions than the T1. This step aligns them using **rigid registration** (translation + rotation only, no deformation):

- Uses SimpleITK's gradient descent optimizer
- Mattes Mutual Information as the similarity metric
- Multi-resolution pyramid (4x, 2x, 1x) for robustness
- After registration, applies the brain mask to zero out non-brain voxels

**Outputs:** `t2w_reg_masked.nii.gz`, `t2f_reg_masked.nii.gz`

#### Step 4: Crop T2w and FLAIR
Applies the same bounding box (computed from the brain mask in Step 2) to crop the registered T2w and FLAIR volumes.

**Outputs:** `t2w_reg_masked_cropped.nii.gz`, `t2f_reg_masked_cropped.nii.gz`

#### Step 5: Stack into 3-Channel Volume
Merges the three cropped modalities (T1, T2w, T2-FLAIR) into a single 4D NIfTI file with shape `(X, Y, Z, 3)`. This is the format expected by the SwinUNETR model (which uses 3 input channels, not 4 like nnU-Net -- T1ce is excluded).

**Output:** `swinbrain_3ch_T1_T2w_T2f.nii.gz`

#### Step 6: Crop Ground Truth Segmentation
Applies the same brain-mask bounding box to crop the raw segmentation mask.

**Output:** `seg_cropped.nii.gz`

#### Step 7: Create Train/Validation Split
Uses scikit-learn's `train_test_split` to split 200 preprocessed cases into:
- **160 training cases** (80%)
- **40 validation cases** (20%)

Saved as `dataset_split.json` with image/label file path pairs.

### About the Pre-trained Encoder (Self-Supervised Learning)

Before fine-tuning on segmentation, the SwinUNETR encoder was pre-trained on **75,861 plain head MRI scans** using a self-supervised learning (SSL) approach (`SwinUNETR_pretrain_DDP.py`). This is important context for understanding why the model works well with limited labeled data.

**How SSL pre-training works:**
1. Take an input 3-channel MRI volume (T1, T2w, FLAIR)
2. Create **two augmented views** by applying random grid dropout (masking out 40% of voxels in a grid pattern) independently to each channel
3. Feed both views through the SwinUNETR model
4. Train with two losses:
   - **Reconstruction loss (L1):** The model must reconstruct the original unmasked image from the masked input
   - **Contrastive loss:** The two views of the same image should produce similar representations (temperature=0.05)
   - **Combined:** `total_loss = recon_loss + contrastive_loss * recon_loss`
5. Trained with Distributed Data Parallel across 8 GPUs for 500 epochs

The resulting pre-trained weights (`SSL_SwinUNETR.pth`) capture general brain anatomy features without needing any tumor labels.

### Fine-tuning for Segmentation (Fine_tune_commented.ipynb)

#### Step 1: Define Transforms
Training transforms include:
- Load NIfTI images
- Ensure channel-first format
- **Remap labels:** Collapse all tumor sub-classes (1, 2, 3, 4) into a single class (1) for binary segmentation
- Reorient to RAS (Right-Anterior-Superior) standard orientation
- Resample to 1mm isotropic spacing
- Scale intensity to [0, 1] range (clipping at 0-2000)
- **Random crop:** Extract 4 random patches of size 128x128x64 per image, balanced around tumor (50% patches centered on tumor, 50% on background)

Validation transforms are the same except no random cropping -- the full volume is kept intact for sliding window inference.

#### Step 2: Create Model
```python
model = SwinUNETR(
    spatial_dims=3,
    in_channels=3,           # T1, T2w, T2-FLAIR
    out_channels=2,          # Background vs Tumor
    feature_size=24,         # Must match pre-trained weights
    use_checkpoint=True,     # Gradient checkpointing to save GPU memory
)
```

The pre-trained SSL weights are loaded into the encoder portion of the model. Keys with `module.` prefix (from DDP training) are cleaned, and only matching layers are loaded (`strict=False` allows the decoder/output heads to initialize randomly).

#### Step 3: Training Loop
- **Loss:** DiceCE Loss (combination of Dice loss + Cross-Entropy loss)
- **Optimizer:** AdamW (lr=1e-4, weight_decay=1e-5)
- **Scheduler:** Cosine annealing over 150 epochs
- **Validation:** Sliding window inference with ROI size 128x128x64, processing 4 patches at a time
- **Metric:** Dice score (excluding background class)
- **Best model saved** whenever validation Dice improves

Training logs show the model improving from ~0.50 Dice at epoch 1 to ~0.58 by epoch 43 (training was still in progress in the notebook).

#### Step 4: Evaluation Visualization
At each validation step, a 3-panel figure is generated showing:
1. Input image (T1 channel, middle axial slice)
2. Ground truth segmentation
3. Model prediction

---

## Additional Component: Classification Pipeline (SwinBrain/)

The `SwinBrain/` directory also contains code for a **Parkinson's Disease (PD) vs. Progressive Supranuclear Palsy (PPS) classification** task using the same pre-trained SwinUNETR encoder:

- **Models.py** defines two classifiers:
  - `ViTClassifier` - Vision Transformer autoencoder with FC classification head
  - `SwinClassifier` - SwinUNETR with adaptive pooling + FC head (24 -> 12 -> 2 classes)
- **finetune_train.py** - Trains the SwinClassifier on PD vs PPS brain MRIs (600 train / 200 val per class, 200 epochs, AdamW + cosine annealing)
- **finetune_test.py** - Evaluates 4 models (ResNet50, DenseNet121, ViT, SwinUNETR) on held-out test data, computing accuracy, F1, precision, and recall

This classification component uses the same 3-channel (T1, T2w, FLAIR) preprocessed MRI inputs resized to 128x128x64.

---

## Project File Structure

```
brain data/
|-- Brain_MRI_segmentation.ipynb       # nnU-Net inference on BraTS2021 glioma
|-- Brain_MRI_segmentation_2.ipynb     # nnU-Net inference on BraTS-MET metastasis
|-- Dataset002_BRATS19/                # Pre-trained nnU-Net model
|   |-- nnUNetTrainer__nnUNetPlans__3d_fullres/
|       |-- dataset.json               # Channel names, labels, training file list
|       |-- dataset_fingerprint.json   # Intensity statistics per channel
|       |-- plans.json                 # Architecture + preprocessing config
|       |-- fold_0/ ... fold_4/        # 5-fold model checkpoints
|           |-- checkpoint_final.pth
|           |-- progress.png           # Training curves
|-- Sample_data/
|   |-- BraTS2021_00495/               # Sample glioma case (T1, T1ce, T2, FLAIR, seg)
|   |-- BraTS-MET-00086-000/           # Sample metastasis case
|-- Transformer_Encoder/               # (in zip files)
    |-- Final_PreProcess_SwinBrain.ipynb  # Full preprocessing pipeline
    |-- Fine_tune_commented.ipynb         # SwinUNETR segmentation fine-tuning
    |-- SwinBrain/
        |-- Models.py                     # ViT and Swin classifier definitions
        |-- finetune_train.py             # PD vs PPS classification training
        |-- finetune_test.py              # Multi-model classification evaluation
        |-- SwinUNETR_pretrain_DDP.py     # SSL pre-training script (multi-GPU)
        |-- dataset_split.json            # Train/val split (160/40 cases)
        |-- Pretrained_models/            # SSL_SwinUNETR.pth weights
```

---

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `nnunetv2` | Self-configuring segmentation framework |
| `nibabel` | Reading/writing NIfTI brain imaging files |
| `medpy` | Dice coefficient and other medical image metrics |
| `SimpleITK` | Image registration (rigid alignment) |
| `monai` | Medical image AI framework (transforms, networks, losses, metrics) |
| `torch` | PyTorch deep learning backend |
| `hd-bet` | Deep learning skull stripping (optional) |

---

## Summary

| Aspect | nnU-Net Pipeline | SwinUNETR Pipeline |
|--------|-----------------|-------------------|
| **Task** | Multi-class tumor segmentation | Binary tumor segmentation |
| **Input channels** | 4 (T1, T1ce, T2, FLAIR) | 3 (T1, T2w, FLAIR) |
| **Output classes** | 5 (BG + edema + necrotic + empty + enhancing) | 2 (BG + tumor) |
| **Architecture** | PlainConvUNet (3D) | SwinUNETR (3D transformer) |
| **Pre-training** | Supervised on BraTS 2019 (80K samples) | Self-supervised on 75K plain head MRIs |
| **What you ran** | Inference only | Full preprocessing + fine-tuning |
| **Best Dice** | 0.966 (on glioma) | ~0.58 (still training, 150 epochs) |
