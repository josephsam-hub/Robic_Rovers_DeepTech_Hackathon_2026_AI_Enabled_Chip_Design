# Dataset Description

## Overview
This dataset contains wafer defect images for semiconductor manufacturing quality control. The dataset includes various types of defects commonly found in wafer fabrication processes.

## Dataset Statistics
- **Total Images**: ~1,517 images
- **Format**: PNG
- **Categories**: 8 defect types + 1 clean class

## Defect Categories

| Category | Number of Images | Description |
|----------|-----------------|-------------|
| Clean | 187 | Non-defective wafer images |
| Bridge Defective | 75 | Bridge defects between circuit lines |
| Crack Defective | 103 | Crack defects on wafer surface |
| LER Defect | 60 | Line Edge Roughness defects |
| Line Collapse Defect | 51 | Collapsed circuit lines |
| LWV Defects | 56 | Line Width Variation defects |
| Open Defects | 50 | Open circuit defects |
| Scratches Defect | 37 | Surface scratches |
| Via Defect | 20 | Via connection defects |

## Data Structure
```
data/Datasets/
├── clean/
├── Bridge Defective/
├── Crack Defective/
├── LER Defect/
├── Line Collapse Defect/
├── LWV defects/
├── Open Defects/
├── Scratches Defect/
└── Via Defect/
```

## Usage
This dataset is used for training deep learning models to automatically detect and classify wafer defects in semiconductor manufacturing.
