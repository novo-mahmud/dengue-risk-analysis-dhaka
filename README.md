


# Dengue Risk Analysis in Dhaka

An independent research project investigating the spatial and infrastructural determinants of Dengue fever in Dhaka, Bangladesh.

**Methods:** Binary Logistic Regression (GLM), Random Forest, Spatial Aggregation.
**Tools:** R (tidyverse, randomForest), RMarkdown.

---

## 📋 Project Overview

This study analyzes patient data from Dhaka to determine whether housing quality (Built Environment) or Geographic Location (Hotspots) is the primary driver of Dengue infection.

### Key Findings
*   **Infrastructure Hypothesis Rejected:** Housing type (Tinshed vs. Building) showed no statistical significance.
*   **Geographic Dominance:** Specific areas (e.g., Sutrapur, Ramna) are significant predictors of infection, indicating localized outbreak clusters.
*   **Administrative Insights:** Aggregated risk analysis identified Sutrapur Thana as the highest risk zone.

---

## 📂 Project Structure

```
.
├── data/
│   └── dengue_dataset.csv    # Raw patient data
├── src/
│   └── dengue_analysis.Rmd   # Reproducible R Code
├── reports/
│   └── dengue_analysis.html  # Full Statistical Report
└── README.md
```

---

## 🚀 How to Run

1.  Clone this repository.
2.  Open `dengue_analysis.Rmd` in RStudio or Posit Cloud.
3.  Run the chunks to reproduce the analysis, plots, and the final report.

---

## 👨‍💻 Author

**Md. Mahmudul Hasan Novo**
*BSc in Civil Engineering, BUET | Data Science Enthusiast*
**Email:** novomahmud@gmail.com

