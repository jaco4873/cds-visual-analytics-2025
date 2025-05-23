# Assignment 4: Detecting Faces in Historical Newspapers

## Overview

This project analyzes the presence of human faces in historical Swiss newspapers over time. Specifically, we examine three historic Swiss newspapers:

- The Journal de Genève (JDG)
- The Gazette de Lausanne (GDL)
- The Impartial (IMP)

The analysis uses a pre-trained CNN model (MTCNN) to detect faces in newspaper page images, groups the results by decade, and visualizes trends in the prevalence of human faces in print media over approximately 200 years.

## Data

This project uses the Swiss newspapers corpus from [this Zenodo dataset](https://zenodo.org/records/3706863). The dataset contains:

- 1008 newspaper pages from the Gazette de Lausanne (1790s-1990s)
- 1982 newspaper pages from the Journal de Genève (1820s-1990s)
- 1634 newspaper pages from the Impartial (1880s-2010s)

The images should placed in `data/newspapers/images/` and organized into three subfolders (GDL, JDG, IMP) corresponding to each newspaper.

## Quickstart

The simplest way to run the assignment is using the provided run.sh script:

```bash
./run.sh
```

Then select option 4 from the menu.

Alternatively, you can run assignment 4 without run.sh as a module:

```bash
# Ensure you are in the right directory (src)
cd src

# Process all newspapers
uv run -m assignment_4.main

# Process only a specific newspaper
uv run -m assignment_4.main --newspaper GDL

```
## Configuration

The project uses a central configuration system in `config.py` that can be modified to customize various aspects:

- Face Detection Configuration: Control MTCNN model parameters including detection thresholds, minimum face size, and scale factor
- Data Configuration: Set data directory paths and specify which newspapers to analyze
- Output Configuration: Define output directory paths for saving results and visualizations

### Command Line Options

Furthermore, a few options are also available via the CLI:

- `--data-dir`: Path to the newspaper images directory (must exist and be a directory)
- `--output-dir`: Path to save the output
- `--newspaper`: Process only a specific newspaper (choices: GDL, JDG, IMP)

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

## Methodology

The analysis follows these steps:

1. Data Loading: Load newspaper images from the dataset.
2. Year Extraction: Extract the year of publication from each filename.
3. Face Detection: Use MTCNN to detect faces in each newspaper page.
4. Decade Aggregation: Group results by decade to analyze temporal trends.
5. Visualization: Create plots showing the percentage of pages with faces over time.

## Results

The analysis produces the following outputs:

1. CSV Files:
   - Individual CSV files for each newspaper showing face count and percentage by decade
   - A combined CSV with results from all newspapers
   - Summary statistics

2. Visualizations:
   - Individual plots for each newspaper showing percentage of pages with faces by decade
   - A comparison plot showing trends across all newspapers

The outputs are saved to the `output/` directory.

Initial detection using default MTCNN thresholds ([0.6, 0.7, 0.7]) produced significant false positives, especially in early decades (1790s-1840s) when photography wasn't yet used in newspapers. This observation led to adjusting the thresholds to [0.7, 0.9, 0.9], substantially increasing the model's strictness. The refined results better align with the historical development of photography and its adoption in print media.

## Analysis

After running our face detection analysis with adjusted thresholds on the three Swiss newspapers, we see more historically accurate trends in how human faces have been represented in print media over the past two centuries.

### Overall Trends

The late 19th century saw newspapers starting the adoption of photography (with faces) in print media. 

Looking at the data chronologically:
- Pre-1850s: Minimal to no faces detected, consistent with the technological limitations of the era
- 1850s-1900s: Gradual introduction of faces, corresponding to early photography adoption in newspapers
- 1900s-1950s: Steady increase as photo reproduction techniques improved
- 1950s-2010s: Dramatic rise, hinting towards a modern visual journalism

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
- The most recent periods show high prevalence - reaching over 65% in the 2000s and exceeding 70% in the 2010s

#### The Gazette de Lausanne (GDL, 1804-1991)

For GDL, the results indicate:

- The earliest periods (1790s-1850s) show minimal face detection (0-3%) which we can interpret as being false positives, as the first photography was used in print media in 1848 ("Photojournalism", Wikipedia 2025).
- No faces detected from 1840s through 1890s
- A clear emergence starting in the 1900s (13%)
- Consistent presence in the 1910s-1930s (10-11%)
- Slight decline in mid-century (8-6%)
- The percentages increase significantly in the 1980s-1990s (20-26%)


#### The Journal de Genève (JDG, 1826-1994)

JDG shows a gradual adoption of facial imagery:

- No faces detected prior to the 1870s
- Initial appearance in the 1870s at very low levels (2%)
- Steady increase through the late 19th century (6% by 1890s)
- Relatively stable presence in early 20th century (4-6%)
- Gradual increase from the 1920s through 1940s (5-11%)
- More significant presence by mid-century (14-15% in 1950s-1960s)
- Reaching peak levels in the 1970s-1980s (around 21%)
- Slight decrease in the 1990s (19.8%)

JDG's pattern shows a more consistent, gradual increase compared to the other newspapers, with fewer dramatic shifts between decades.

### Historical Context and Significance

These patterns reflect several important developments:

1. Technological Evolution: The introduction of photography made it increasingly feasible and affordable to include images in newspapers, clearly visible in the minimal detection before 1870s and gradual increase thereafter ("Photojournalism", Wikipedia 2025).

2. Modern Visual Culture: The dramatic increase in the most recent decades (particularly for IMP exceeding 70% by the 2010s) mirrors broader cultural shifts toward visual media 

3. Regional Differences: The significant difference between IMP and the other two newspapers might reflect different editorial philosophies or regional preferences. IMP shows a notable early adoption in the 1910s-1920s (20%) and dramatically accelerated use by the 1950s (36%), while GDL and JDG followed more conservative trajectories.


It's worth noting that our dataset shows an uneven distribution of pages across decades, with more pages available from recent decades. This reflects the changing volume of newspaper production over time but may also influence the reliability of our decade-to-decade comparisons.

## Limitations

This analysis has several limitations to consider:

1. Model Training Data: MTCNN was only trained on modern photographs, making it prone to misdetecting other objects as faces in historical materials. Other face detection algorithms might perform differently on historical materials.
2. Image Quality: Historical newspaper scans vary in quality, affecting face detection accuracy.
3. Temporal Distribution: Uneven distribution of pages across time periods may affect decade-to-decade comparisons.
4. Detection Thresholds: Analysis required adjusted confidence thresholds ([0.7, 0.9, 0.9] vs default [0.6, 0.7, 0.7]) to reduce false positives. This adjustment was based on visual inspection rather than systematic optimization.
5. Qualitative Analysis: Our analysis counts faces but doesn't distinguish between photographs, illustrations, or engravings, nor considers face size, placement, or prominence.
6. Lack of Ground Truth: Without manually annotated historical newspaper data, we couldn't quantitatively evaluate detection performance across different eras.

## Credits

This project was created as part of the Cultural Data Science - Visual Analytics course at Aarhus University.

- Dataset: Swiss newspapers corpus from [Zenodo](https://zenodo.org/records/3706863)
- Face detection: [FaceNet-PyTorch](https://github.com/timesler/facenet-pytorch) implementation of MTCNN
- Note in MTCNN training on modern photographs: [FaceNet: A unified embedding for face recognition and clustering](https://ieeexplore.ieee.org/document/7298682)
- First photography in newspaper: [Wikipedia](https://en.wikipedia.org/wiki/Photojournalism)
