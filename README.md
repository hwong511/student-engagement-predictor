# Student Engagement Detection from Digital Learning Analytics

**Automated prediction of BROMP-coded behavioral engagement from IMS Caliper event logs in Carnegie Learning's MATHStream adaptive mathematics platform.**

---

## Research Question

**Can we automate expensive human behavioral coding using only digital trace data?**

This project investigates whether digital learning analytics (IMS Caliper event logs) can reliably predict human-coded behavioral engagement observations (BROMP protocol), with the goal of reducing the cost and scale limitations of traditional classroom observation methods in adaptive learning environments.

---

## Quick Start

### Prerequisites

**R packages:**
```r
install.packages(c("readr", "dplyr", "lubridate", "tidyr", 
                   "janitor", "fuzzyjoin", "data.table", "here"))
```

**Python packages:**
```bash
pip install -r requirements.txt
```

### Running the Complete Pipeline

**Option 1: Production Scripts (Recommended)**

```bash
# Step 0: Clean raw data (requires raw Excel files - DUA protected)
Rscript scripts/00_clean_data.R

# Step 1: Merge BROMP and Caliper data with fuzzy time windows
Rscript scripts/01_merge_data.R

# Step 2: Exploratory stream normalization (optional)
Rscript scripts/02_stream_EDA.R

# Step 3: Train XGBoost model (with caching for faster iteration)
python scripts/03_train_model.py
```

**Option 2: Interactive Notebooks (Exploratory)**

```bash
# R notebooks (using Quarto)
quarto render notebooks/00_clean_data.qmd
quarto render notebooks/01_data_merge_bromp\&caliper.qmd
quarto render notebooks/02_caliper_rq_norm_stream.qmd

# Python notebooks (using Jupyter)
jupyter notebook notebooks/03_xgboost.ipynb
```

### Expected Runtime

- **Data cleaning:** ~1 minute
- **Data merging:** ~2-3 minutes
- **Model training (first run):** ~4-5 hours (grid search)
- **Model training (cached):** ~1 minute (loads cached results)

To re-run grid search, delete: `outputs/cache/grid_search_results.pkl`

---

## Project Structure

```
student-engagement-predictor/
├── data/                          # Data directory (DUA-protected)
│   ├── README.md                  # Data documentation & privacy notice
│   ├── BROMP-clean.csv            # Human-coded behavioral observations
│   ├── caliper-bromp-data-2.xlsx  # Raw Caliper event logs
│   └── BROMP_sorted_by_studentid.xlsx  # Raw BROMP observations
│
├── scripts/                       # Production R & Python scripts
│   ├── 00_clean_data.R            # Data cleaning (Excel → CSV)
│   ├── 01_merge_data.R            # Merge BROMP + Caliper with fuzzy time windows
│   ├── 02_stream_EDA.R            # Exploratory analysis (optional)
│   └── 03_train_model.py          # XGBoost training with grid search
│
├── notebooks/                     # Interactive exploratory notebooks
│   ├── 00_clean_data.qmd          # Quarto notebook (R) - data cleaning
│   ├── 01_data_merge_bromp&caliper.qmd  # Quarto (R) - data merging
│   ├── 02_caliper_rq_norm_stream.qmd    # Quarto (R) - stream normalization
│   └── 03_xgboost.ipynb           # Jupyter (Python) - model analysis
│
├── src/                           # Shared Python utilities
│   ├── __init__.py
│   └── utils.py                   # Helper functions (load data, create lags, save models)
│
├── outputs/                       # Generated results (auto-created)
│   ├── cache/                     # Grid search cache
│   └── figures/data/              # Visualization-ready CSV files
│
├── tests/                         # Test directory
├── docs/                          # Documentation directory
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## Results & Analysis

### Model Performance (Test Set)

**Classification Metrics:**
- AUC: 0.6385 (vs. benchmark: 0.71)
- F1 Score: 0.7975
- Accuracy: 0.70 (vs. baseline: 0.60)

**Classification Report:**
|              | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| DISENGAGED   | 0.43      | 0.42   | 0.42     | 201     |
| ENGAGED      | 0.80      | 0.80   | 0.80     | 567     |
| **Accuracy** |           |        | **0.70** | 768     |
| Macro Avg    | 0.61      | 0.61   | 0.61     | 768     |
| Weighted Avg | 0.70      | 0.70   | 0.70     | 768     |

**Validation vs. Test:**
- Minimal overfitting (train-test gap: <0.05 AUC)
- Consistent performance across student-level splits
- Generalization to unseen students maintained

---

## Research Context

### Theoretical Framework

This work builds on foundational research in **Educational Data Mining (EDM)** and **Learning Analytics:**

**Behavioral Coding:**
- BROMP (Baker-Rodrigo Observation Method Protocol)
- Standardized human observation of student engagement
- Gold standard but expensive and time-intensive

**Automated Detection History:**
- Baker et al. (2004): First automated detector (~0.60 AUC)
- Baker et al. (2012): Improved detector with temporal features (0.71 AUC)
- This work (2024): Comprehensive feature engineering (0.639 AUC)

**Performance Plateau:**
Despite advances in ML techniques (XGBoost vs. logistic regression) and richer feature engineering, performance has remained relatively stable over 20 years. This suggests **fundamental limitations** in what digital signals can reveal about cognitive engagement.

### Implications for Educational Technology

**1. Automated Engagement Detection Has Fundamental Limits**

Even with:
- Modern ML algorithms (XGBoost)
- Rich temporal features (41 engineered features)
- Proper validation methodology (student-level CV)
- Individual normalization (personalized baselines)

...we cannot reliably distinguish all engagement states from digital logs alone.

**2. Screening vs. Replacement**

Automated detection is better suited for:
- **Large-scale screening:** Flag students for human follow-up
- **Trend analysis:** Track engagement patterns over time
- **Intervention triggers:** Detect sustained disengagement

Not suitable for:
- **Replacing human observers:** Too many false positives/negatives
- **High-stakes decisions:** Insufficient accuracy for individual assessment
- **Formative feedback:** Cannot distinguish subtle engagement nuances

**3. Value of Multimodal Data**

Future work should incorporate:
- Eye-tracking data (attention allocation)
- Facial expression analysis (affect detection)
- Physiological sensors (arousal, stress)
- Audio analysis (collaboration patterns)

Digital logs alone are insufficient for comprehensive engagement monitoring.

---

## References

### Foundational Work

**BROMP Methodology:**
- Baker, R. S., et al. (2004). *Detecting student misuse of intelligent tutoring systems.* Proceedings of ITS 2004.
- Ocumpaugh, J., Baker, R., & Rodrigo, M. M. (2015). *Baker Rodrigo Observation Method Protocol (BROMP) 1.0. Training Manual version 1.0.*

**Automated Engagement Detection:**
- Baker, R. S., et al. (2012). *Towards sensor-free affect detection in cognitive tutor algebra.* Proceedings of EDM 2012.
- Baker, R. S., & Rossi, L. M. (2013). *Assessing the disengaged behavior of learners.* Handbook of Research on Educational Communications and Technology.

### Technical Standards

**IMS Caliper Analytics:**
- IMS Global Learning Consortium. (2016). *Caliper Analytics Specification v1.1.*
- Event taxonomy for digital learning platforms

**Adaptive Learning Platforms:**
- Carnegie Learning MATHStream Platform Documentation
- Video-based adaptive mathematics instruction

### Related Research

**Learning Analytics:**
- Educational Data Mining (EDM) Community
- Society for Learning Analytics Research (SoLAR)
- Journal of Educational Data Mining

**Temporal Data Mining:**
- Time-series feature engineering in educational contexts
- Behavioral pattern detection in learning systems

---

## Acknowledgments

- **Carnegie Learning** for data access under Data Use Agreement
- **PhD Advisor:** Sachi Sanghavi
- **Baker Research Group:** For foundational BROMP methodology
- **IMS Global:** Caliper Analytics specification
- **XGBoost Community:** Machine learning framework
- **R & Python Communities:** Open-source data science tools

---

## License & Data Use

### Code License

**MIT License** - See [LICENSE](LICENSE) file for details.

The code in this repository is freely available for use, modification, and distribution.

### Data Restrictions

**Proprietary Data - NOT INCLUDED**

The data files (`BROMP-clean.csv`, `caliper-bromp-data-2.xlsx`, etc.) are subject to Data Use Agreement with Carnegie Learning and **cannot be shared publicly** due to:

- Student privacy protections (FERPA compliance)
- Carnegie Learning intellectual property rights
- Institutional Review Board (IRB) restrictions

**For Collaborators:**
- Contact repository owner for data access under DUA
- Requires institutional approval and signed agreement

**For External Users:**
- Code and methodology provided for transparency
- Apply techniques to your own educational datasets
- Cannot reproduce exact results without Carnegie Learning data

---

## Contact

**Ho Wong**  
EMAIL: [howong112@outlook.com]  
LINKEDIN: [http://linkedin.com/in/hwong511]  
PORTFOLIO: [http://hwong511.github.io]
