# ImageX

ImageX Brain MRI encoder and CNN decoder project.

## Overview

This project implements a Brain MRI image processing pipeline using encoder-decoder CNN architecture.

## Dataset Setup

This project supports two types of datasets:

### 1. **Small Datasets (Local)**
For datasets that can be directly downloaded and stored locally:
- Place them in the `data/local/` directory
- See `data/local/README.md` for structure guidelines

### 2. **Large Datasets (Google Drive)**
For large datasets that cannot be easily downloaded:
- Access via Google Drive links
- See `DATASETS.md` for all available datasets and their Google Drive links

## Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/sivatej800-oss/ImageX.git
   cd ImageX
   ```

2. **Set up your datasets**
   - Check `DATASETS.md` for available datasets
   - Download small datasets to `data/local/`
   - Access large datasets via the Google Drive links provided

3. **Configure data paths**
   - Update your code to point to the appropriate data directory
   - For local datasets: `data/local/your_dataset/`
   - For Google Drive: Mount or download from the links in `DATASETS.md`

## Documentation

- **[DATASETS.md](DATASETS.md)** - Complete information about all datasets, including Google Drive links for large datasets

## Contributing

When adding new datasets:
- Small datasets (< 100MB): Add to `data/local/` and document in `DATASETS.md`
- Large datasets: Upload to Google Drive, add the link to `DATASETS.md`

## Project Structure

```
ImageX/
├── data/
│   └── local/          # Local datasets (git-ignored)
├── DATASETS.md         # Dataset documentation and Google Drive links
└── README.md           # This file
```

