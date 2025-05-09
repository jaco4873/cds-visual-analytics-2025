# Assignment 4: Detecting Faces in Historical Newspapers

## Overview

This project analyzes the presence of human faces in historical Swiss newspapers over time. Specifically, we examine three historic Swiss newspapers:

- The Journal de Genève (JDG, 1826-1994)
- The Gazette de Lausanne (GDL, 1804-1991)
- The Impartial (IMP, 1881-2017)

The analysis uses a pre-trained CNN model (MTCNN) to detect faces in newspaper page images, groups the results by decade, and visualizes trends in the prevalence of human faces in print media over approximately 200 years.

## Project Structure

```
assignment_4/
├── config.py                  # Configuration settings 
├── main.py                    # Main script to run the analysis
├── services/                  # Service modules
│   ├── data_service.py        # Data loading and processing
│   └── face_detection_service.py  # Face detection with MTCNN
├── utils/                     # Utility modules
│   └── visualization.py       # Visualization utilities
└── output/                    # Output directory
    ├── results/               # CSV results
    └── plots/                 # Visualization plots
```

## Requirements

This project requires the following Python packages:

- torch
- facenet-pytorch
- PIL
- pandas
- matplotlib
- numpy
- pydantic-settings

## Installation

Clone the repository and install the required packages:

```bash
# Clone the repository
git clone <repository-url>
cd <repository-directory>

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run the main script to process the newspaper images and generate the analysis:

```bash
# Process all newspapers
python -m src.assignment_4.main

# Process only a specific newspaper
python -m src.assignment_4.main --newspaper GDL

# Specify custom data and output directories
python -m src.assignment_4.main --data_dir /path/to/data --output_dir /path/to/output
```

## Data

This project uses the Swiss newspapers corpus from [this Zenodo dataset](https://zenodo.org/records/3706863). The dataset contains:

- 100 newspaper pages from the Journal de Genève (1826-1994)
- 100 newspaper pages from the Gazette de Lausanne (1804-1991)
- 100 newspaper pages from the Impartial (1881-2017)

The images are located at `data/newspapers/images/images/` and organized into three folders (GDL, JDG, IMP) corresponding to each newspaper.

## Methodology

The analysis follows these steps:

1. **Data Loading**: Load newspaper images from the dataset.
2. **Year Extraction**: Extract the year of publication from each filename.
3. **Face Detection**: Use MTCNN to detect faces in each newspaper page.
4. **Decade Aggregation**: Group results by decade to analyze temporal trends.
5. **Visualization**: Create plots showing the percentage of pages with faces over time.

## Results

The analysis produces the following outputs:

1. **CSV Files**:
   - Individual CSV files for each newspaper showing face count and percentage by decade
   - A combined CSV with results from all newspapers
   - Summary statistics

2. **Visualizations**:
   - Individual plots for each newspaper showing percentage of pages with faces by decade
   - A comparison plot showing trends across all newspapers

The outputs are saved to the `output/` directory.

## Interpretation

The results of this analysis demonstrate how the presence of human faces in printed media has evolved over time. These changes likely reflect:

1. **Technological Evolution**: The development of photography and printing technology
2. **Cultural Shifts**: Changes in journalistic practices and media presentation
3. **Social Factors**: The increasing personalization of news and media content

A detailed interpretation of the results is provided in the analysis section below.

## Analysis

(Note: This section would contain the actual interpretation of results after running the analysis)

## Limitations

This analysis has several limitations to consider:

1. **Sample Size**: The dataset contains only 100 pages per newspaper, which might not fully represent all publication periods.
2. **Image Quality**: Historical newspaper scans vary in quality, which may affect face detection accuracy.
3. **Face Detection Accuracy**: The MTCNN model may have different detection rates for different image qualities and historical periods.
4. **Historical Context**: Changes in newspaper formats, layouts, and content focus over time might influence the results.

## Credits

This project was created as part of the Cultural Data Science - Visual Analytics course at Aarhus University.

- Dataset: Swiss newspapers corpus from [Zenodo](https://zenodo.org/records/3706863)
- Face detection: [FaceNet-PyTorch](https://github.com/timesler/facenet-pytorch) implementation of MTCNN

## License

This project is licensed under the MIT License.
