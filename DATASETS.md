# Datasets for ImageX

This document describes the datasets used in the ImageX Brain MRI encoder and CNN decoder project.

## Dataset Organization

This project uses two types of dataset references:

### 1. Local Datasets (Small, Downloadable)
These datasets can be directly downloaded and placed in the repository.

**Location:** `data/local/`

**Instructions:**
1. Download the dataset
2. Extract/place it in the `data/local/` directory
3. Follow the structure described in `data/local/README.md`

**Available Datasets:**
- *Add your downloadable datasets here with download instructions*

Example:
```
Dataset Name: Brain MRI Small Sample
Description: Small sample dataset for testing
Size: 50 MB
Location: data/local/brain_mri_sample/
Download: [Add download link if applicable]
```

### 2. Google Drive Linked Datasets (Large)
These datasets are too large to include in the repository and are hosted on Google Drive.

**Instructions:**
1. Access the Google Drive link provided below
2. Download or mount the Google Drive folder
3. Update your code to point to the downloaded location or use Google Colab to access directly

**Available Datasets:**

#### Brain MRI Full Dataset
- **Description:** Complete Brain MRI dataset for training
- **Size:** Large (multiple GB)
- **Google Drive Link:** [Add your Google Drive link here]
- **Contents:** 
  - Training images
  - Validation images
  - Test images
  - Labels/annotations
- **Usage:** 
  ```python
  # Example: Update the data path in your code
  DATA_PATH = "/path/to/downloaded/dataset"
  # Or if using Google Colab:
  # from google.colab import drive
  # drive.mount('/content/drive')
  # DATA_PATH = "/content/drive/MyDrive/brain_mri_dataset"
  ```

#### Additional Large Dataset
- **Description:** [Add description]
- **Size:** [Add size]
- **Google Drive Link:** [Add your Google Drive link here]
- **Usage:** [Add usage instructions]

## How to Add Your Datasets

### For Small/Downloadable Datasets:
1. Add the dataset to `data/local/`
2. Update this file with dataset information in the "Local Datasets" section

### For Large/Google Drive Datasets:
1. Upload your dataset to Google Drive
2. Make the folder shareable (get a shareable link)
3. Add the dataset information in the "Google Drive Linked Datasets" section above
4. Include:
   - Dataset name
   - Description
   - Size
   - Google Drive link
   - Usage instructions

## Notes

- The `data/local/` directory is git-ignored to prevent committing large files
- Always verify Google Drive links are accessible before sharing
- Consider using Google Colab for direct Google Drive access when working with large datasets
