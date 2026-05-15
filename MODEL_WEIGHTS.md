# Model weights

Model checkpoints are not committed to this repository (size and licensing). Download them from the sources below and place them where indicated.

## 1. SSL pre-trained SwinUNETR encoder

Used by [`notebooks/02_finetune_swinunetr.ipynb`](notebooks/02_finetune_swinunetr.ipynb) to warm-start the tumor segmentation fine-tuning.

- **File:** `SSL_SwinUNETR.pth` (~67 MB)
- **Download:** [Google Drive](https://drive.google.com/file/d/1vXdoj8LErVO_jgDaKauBa3zISVW9v-m1/view?usp=drive_link)
- **Source:** [MAI-Lab West China Hospital — SwinBrain](https://github.com/MAI-Lab-West-China-Hospital/SwinBrain)
- **Place at:** `weights/SSL_SwinUNETR.pth` (create the `weights/` directory if needed)
- **Training:** 75,861 plain head MRI volumes (T1w, T2w, FLAIR), 500 epochs on 8 GPUs, masked reconstruction (L1) + contrastive loss.

## 2. nnU-Net v2 — Dataset002_BRATS19

Used by [`notebooks/03_nnunet_inference_glioma.ipynb`](notebooks/03_nnunet_inference_glioma.ipynb) and [`04_nnunet_inference_metastasis.ipynb`](notebooks/04_nnunet_inference_metastasis.ipynb).

- **Files:** 5-fold model checkpoints + `plans.json`, `dataset.json`, `dataset_fingerprint.json` (~1.3 GB total)
- **Source:** nnU-Net v2 pretrained model zoo, dataset name `Dataset002_BRATS19`. See [MIC-DKFZ/nnUNet — pretrained models](https://github.com/MIC-DKFZ/nnUNet#how-to-get-pretrained-models).
- **Place at:** any directory you want, then point the `nnUNet_results` environment variable at it before running the notebooks.

## 3. Fine-tuned SwinUNETR (your output)

Produced by running [`notebooks/02_finetune_swinunetr.ipynb`](notebooks/02_finetune_swinunetr.ipynb) on BraTS data.

- **File:** `best_metric_model.pth` (~67 MB)
- **Place at:** `weights/best_metric_model.pth`
- Saved automatically each time validation Dice improves.

---

## BraTS data

This repository does not redistribute BraTS imaging data. Request access at:

- [BraTS 2021](https://www.synapse.org/#!Synapse:syn25829067/wiki/610863) (glioma)
- [BraTS-MET](https://www.synapse.org/#!Synapse:syn51156910/wiki/) (metastasis)

After downloading, update the file paths in [`configs/dataset_split.json`](configs/dataset_split.json) (currently set to Google Colab Drive paths used during development).
