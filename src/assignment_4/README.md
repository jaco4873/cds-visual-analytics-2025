# Assignment 4: Detecting Faces in Historical Newspapers

## Overview

This project analyzes the presence of human faces in historical Swiss newspapers over time. Specifically, we examine three historic Swiss newspapers:

- The Journal de Genève (JDG)
- The Gazette de Lausanne (GDL)
- The Impartial (IMP)

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

## Getting Started

Clone the repository and install the required packages:

```bash
# Clone the repository
git clone <repository-url>
cd <repository-directory>

# Install dependencies
uv sync
```

## Usage

Run the main script to process the newspaper images and generate the analysis:

The simplest way to run the assignment is using the provided run.sh script:
```bash
./run.sh
```
Then select option 4 from the menu.

Alternatively, you can run assignment 4 as a module as below:

```bash
# Ensure you are in the right directory (src)
cd src

# Process all newspapers
uv run -m assignment_4.main

# Process only a specific newspaper
uv run -m assignment_4.main --newspaper GDL

# Specify custom data and output directories
uv run -m assignment_4.main --data-dir /path/to/data --output-dir /path/to/output
```

### Command Line Options

- `--data-dir`: Path to the newspaper images directory (must exist and be a directory)
- `--output-dir`: Path to save the output
- `--newspaper`: Process only a specific newspaper (choices: GDL, JDG, IMP)

## Data

This project uses the Swiss newspapers corpus from [this Zenodo dataset](https://zenodo.org/records/3706863). The dataset contains:

- 1008 newspaper pages from the Gazette de Lausanne (1790s-1990s)
- 1982 newspaper pages from the Journal de Genève (1820s-1990s)
- 1634 newspaper pages from the Impartial (1880s-2010s)

The images are located at `data/newspapers/images/` and organized into three folders (GDL, JDG, IMP) corresponding to each newspaper.

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

Initial detection using default MTCNN thresholds ([0.6, 0.7, 0.7]) produced significant false positives, especially in early decades (1790s-1840s) when photography wasn't yet used in newspapers. This observation led to adjusting the thresholds to [0.7, 0.9, 0.9], substantially increasing the model's strictness. The refined results better align with the historical development of photography and its adoption in print media.

## Analysis

After running our face detection analysis with adjusted thresholds on the three Swiss newspapers, we see more historically accurate trends in how human faces have been represented in print media over the past two centuries.

### Overall Trends

The late 19th century saw newspapers gradually containing more and more faces, mirroring the technologic development in society where photography became more and more common

Looking at the data chronologically:
- Pre-1850s: Minimal to no faces detected, consistent with the technological limitations of the era
- 1850s-1900s: Gradual introduction of faces, corresponding to early photography adoption in newspapers
- 1900s-1950s: Steady increase as photo reproduction techniques improved
- 1950s-2010s: Dramatic rise reflecting modern visual journalism

![Newspaper Comparison](output/plots/newspaper_comparison.png)

As shown in the comparison visualization above,  GDL and JDG followed similar trajectories in adoptation throughout the decades, where as IMP rapidly increased the use of faces from the 1950's and up until the most recent decades included faces in their newspapers significantly more than GDL and JDG. It's also worth noting that we only have data for GDL and JDF up to the 1990's. A search on Google showed that these two newspapers in fact merged in 1991 and continued under a new name, which is a good explanation as to why we don't have any data after that.

### Individual Newspaper Patterns

#### The Impartial (IMP, 1881-2017)

IMP shows a distinctive pattern:
- Starting with 0% in the 1880s, faces appear in around 8% of pages in the 1890s
- Early 1900s shows relatively low percentages (5-6%)
- A significant jump occurs in the 1910s-1920s, reaching around 20%
- The percentage stabilizes around 18-22% through the 1930s
- A noticeable jump to 36% in the 1950s
- After some fluctuation in mid-century, there's a dramatic increase in the late 20th/early 21st century
- The most recent periods show extraordinary prevalence - reaching over 65% in the 2000s and exceeding 70% in the 2010s

#### The Gazette de Lausanne (GDL, 1804-1991)

For GDL, the results indicate:
- The earliest periods (1790s-1850s) show minimal face detection (0-3%) which we can reasonable interpret as being mostly false positives despite we adjusted the detection thresholds upwards from the defaults. 
  - We can consider these false positives as we now photography was not used before the last part of the 19th century.
- No faces detected from 1840s through 1890s
- A clear emergence starting in the 1900s (13%)
- Consistent presence in the 1910s-1930s (10-11%)
- Slight decline in mid-century (8-6%)
- The percentages increase significantly in the 1980s-1990s (20-26%)

These results give reason to wonder how reliable the face detection algorithms is throughout, and whether the chosen detection thresholds are too restrictive.

#### The Journal de Genève (JDG, 1826-1994)

JDG demonstrates a gradual adoption of facial imagery:
- No faces detected prior to the 1870s
- Initial appearance in the 1870s at very low levels (2%)
- Steady increase through the late 19th century (6% by 1890s)
- Relatively stable presence in early 20th century (4-6%)
- Gradual increase from the 1920s through 1940s (5-11%)
- More significant presence by mid-century (14-15% in 1950s-1960s)
- Reaching peak levels in the 1970s-1980s (around 21%)
- Slight decrease in the 1990s (19.8%)

JDG's pattern shows a more consistent, gradual increase compared to the other newspapers, with fewer dramatic shifts between decades.
From a non-domain experts perspective, this gradual increase seems plausible and speaks in favour of the model configuration.

### Historical Context and Significance

These patterns reflect several important developments:

1. **Technological Evolution**: The introduction of photography and improvements in printing technology made it increasingly feasible and affordable to include images in newspapers, clearly visible in the minimal detection before 1870s and gradual increase thereafter.

2. **Cultural Transformation**: The increasing prevalence of faces reflects a fundamental shift in how news was presented - from dense text toward pages with a stronger visual appeal.

3. **Regional Differences**: The significant difference between IMP and the other two newspapers might reflect different editorial philosophies or regional preferences. IMP shows a notable early adoption in the 1910s-1920s (20%) and dramatically accelerated use by the 1950s (36%), while GDL and JDG followed more conservative trajectories.

4. **Modern Visual Culture**: The dramatic increase in the most recent decades (particularly for IMP exceeding 70% by the 2010s) mirrors broader cultural shifts toward visual media and personality-focused journalism.

It's worth noting that our dataset shows an uneven distribution of pages across decades, with more pages available from recent decades. This reflects the changing volume of newspaper production over time but may also influence the reliability of our decade-to-decade comparisons.

This analysis provides a quantitative glimpse into the visual transformation of print media, showing how newspapers evolved from text-heavy documents to increasingly visual platforms that prominently feature human faces. The trend accelerated dramatically in the late 20th and early 21st centuries, suggesting a fundamental shift in how information is communicated and consumed.

## Limitations

This analysis has several limitations to consider:

1. **Image Quality**: Historical newspaper scans vary in quality, which may affect face detection accuracy.
2. **Historical Context**: Changes in newspaper formats, layouts, and content focus over time might influence the results.
3. **Temporal Distribution**: While we have a substantial number of pages from each newspaper, the distribution across different time periods may not be even, potentially affecting decade-to-decade comparisons.
4. **Detection Thresholds**: The analysis required adjusting detection confidence thresholds ([0.7, 0.9, 0.9] instead of default [0.6, 0.7, 0.7]) to reduce false positives in earlier periods. 
This adjustment was necessary because the MTCNN model, trained primarily on modern photographs, can misidentify illustrations or decorative elements in historical newspapers as faces. This highlights the challenges of applying modern computer vision tools to historical materials.
1. **Limited Hyperparameter Optimization**: Our threshold adjustment ([0.7, 0.9, 0.9]) was based on visual inspection rather than systematic optimization. A more rigorous approach using grid search with manually annotated ground truth data from different periods could potentially yield better discrimination between real faces and false positives.
2. **Model Selection**: We exclusively used MTCNN for face detection. Other modern face detection algorithms might perform differently on historical materials and could be compared in future work.
3. **Qualitative Analysis**: Our analysis counts faces but doesn't distinguish between photographs, illustrations, or engravings, nor does it consider face size, placement, or prominence on the page - all factors that influence visual impact.
4. **Lack of Ground Truth**: Without manually annotated historical newspaper data, we couldn't quantitatively evaluate detection performance (precision/recall) across different eras, making it difficult to objectively measure false positive and false negative rates.

## Requirements

This project requires the following Python packages:

- torch
- facenet-pytorch
- PIL
- pandas
- matplotlib
- numpy
- pydantic-settings
- click

## Credits

This project was created as part of the Cultural Data Science - Visual Analytics course at Aarhus University.

- Dataset: Swiss newspapers corpus from [Zenodo](https://zenodo.org/records/3706863)
- Face detection: [FaceNet-PyTorch](https://github.com/timesler/facenet-pytorch) implementation of MTCNN


