# Drug Category Prediction System

A Neuro-Symbolic (NeSy) system for predicting drug therapeutic categories (ATC codes) based on brain gene expression and drug target information. The system provides explainable predictions with drug similarity matching and CNS activity classification.

> **⚠️ Important Notes:**
> - This is a **lecture project**. Please read the [project report PDF](DrugCategoryPredication_and_Explanation%20(3).pdf) before using the system.
> - **Dataset Access**: Download the required datasets from [Google Drive](https://drive.google.com/drive/u/1/folders/1rnPldk5j06xEjuf3Tqk_dHZKW6ji0HOq?usp=sharing)
> - **Docker Users**: Check `.dockerignore` before running. Some files are excluded to reduce Docker image size. You may need to adjust it based on your requirements.

## Quick Start

### Prerequisites
- Python 3.10+ (3.11+ recommended for Docker)
- Docker & Docker Compose (for containerized deployment)
- Dataset: DrugBank XML, gene expression data (available on Google Drive - see link above)

### Run Locally (CLI)

```bash
# Activate virtual environment
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate   # Linux/Mac

# Run interactive CLI
python interactive_predictor.py
```

Enter target genes (e.g., `APOE, MAPT, CSF3R`) and get predictions with explanations.

### Run Locally (Web Interface)

```bash
# Terminal 1: Start API server
python api_server.py
# Server starts at http://localhost:5000

# Terminal 2: Open web interface in browser
# Navigate to http://localhost:5000
```

### Run with Docker Compose
Open Docker

```bash
docker-compose up
# Access web interface at http://localhost:5000
```

---

## System Architecture

### Pipeline Stages
1. **Gene Expression Processing**: Analyzes single-cell RNA-seq data for brain cell clusters
2. **Drug Target Extraction**: Extracts genes targeted by drugs (DrugBank + DRUGseqr)
3. **ATC Hierarchy Assignment**: Maps drugs to therapeutic categories
4. **Feature Integration**: Combines gene expression and drug data into feature vectors
5. **Model Training**: Trains CNS classifier and category classifier
6. **Prediction & Explanation**: Predicts categories and provides reasoning

### Key Components

| Component | Purpose |
|-----------|---------|
| **interactive_predictor.py** | CLI interface for gene input and prediction |
| **api_server.py** | Flask backend for web interface |
| **src/agents/sklearn_agent.py** | ML model inference + drug similarity search |
| **src/agents/evidence_agent.py** | Explainability & reasoning engine |
| **web/index.html** | Web UI for predictions |

---

## Testing the System

### Example Gene Sets

Test the system with these well-known gene sets:

**Example 1: Neurodegenerative Disease Genes**
```
APOE, MAPT, SNCA, LRRK2, GBA
```
Expected: CNS-active drugs, nervous system categories

**Example 2: Inflammatory Response Genes**
```
TNF, IL6, IL1B, NF-KAPPA-B, CSF3R, ELANE
```
Expected: Immunomodulating agents, antiinfectives

**Example 3: Cardiovascular Genes**
```
LDLR, APOB, HMGCR, ACE, AGTR1
```
Expected: Cardiovascular system drugs

**Example 4: Cancer-Related Genes**
```
TP53, BRCA1, MYC, HER2, EGFR
```
Expected: Antineoplastic agents, immunomodulating drugs

### CLI Usage Example

```bash
(.venv) > python interactive_predictor.py

Enter target genes (comma-separated): APOE, MAPT, CSF3R
Enter drug name (optional): Test_Drug

--- PREDICTION RESULTS ---
Predicted Category: N (Nervous System)
Confidence: 0.87

Brain Evidence:
- BES: 0.92 (High brain expression)
- BSR: 2.34 (High brain specificity)
- CNS Score: 0.85 (CNS-active)

Similar Drugs:
- Donepezil (overlap: 2)
- Tacrine (overlap: 1)

Evidence: Strong brain expression in multiple clusters...
```

---

## Training the Full System

To retrain the complete Neuro-Symbolic pipeline from scratch, use the `nesy_system.py` orchestrator:

### Complete Training Pipeline

```bash
# Activate virtual environment
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate   # Linux/Mac

# Run the complete NeSy system
python nesy_system.py
```

### What the Pipeline Does

The `nesy_system.py` script executes all 6 stages of the NeSy pipeline sequentially:

| Stage | Task | Input | Output | Duration |
|-------|------|-------|--------|----------|
| 1 | Brain Gene Expression | scRNA cluster data | Expression profiles by brain region | ~1-2 min |
| 2 | Extract Drug Targets | DrugBank XML + DRUGseqr | Drug-gene relationships | ~2-3 min |
| 3 | Fetch ATC Hierarchy | Extracted drug data | Therapeutic category mappings | ~1 min |
| 4 | Feature Integration | Gene + drug data | Integrated feature vectors | ~2-3 min |
| 5a | CNS Classifier | Feature vectors | CNS drug classifier model | ~1-2 min |
| 5b | Category Classifier | Feature vectors | Drug category classifier model | ~2-3 min |
| 6 | Predict & Evaluate | Test data | Predictions + evaluation metrics | ~1 min |

**Total Pipeline Time**: ~10-15 minutes (depends on data volume and hardware)

### Output Files

After successful execution, the following files are generated in `results/`:

```
results/
├── task1_brain_expression.csv           # Brain gene expression profiles
├── task2_drug_targets.csv               # Extracted drug target genes
├── task3_atc_hierarchy.csv              # Drug to ATC category mappings
├── task4_integrated_features.npy        # Feature matrix for ML models
├── task4_valid_indices.npy              # Indices of valid samples
├── task5_model_comparison.csv           # CNS classifier comparison
├── task5a_model_comparison.csv          # Category classifier comparison
├── task6_predictions.csv                # Model predictions on test set
├── task6_classification_report.csv      # Precision/recall/F1 metrics
├── task6_example_predictions.csv        # Sample predictions with explanations
└── EVALUATION_METRICS.json              # Complete metrics summary
```

Models are saved in `models/`:
```
models/
├── cns_classifier.joblib                # Trained CNS classifier
└── (Category classifier saved in results/)
```

### Example Training Run Output

```
================================================================================
NEURO-SYMBOLIC DRUG CATEGORY PREDICTION SYSTEM
Following gene-correlation.json Design Pattern
================================================================================

================================================================================
STEP 0: DATASET VALIDATION
================================================================================
  ✓ DrugBank XML loaded: 13,410 drugs
  ✓ RNA cluster data loaded: 19,574 genes × 34 cell clusters
  ✓ DRUGseqr.gmt found: dataset/DRUGseqr.gmt
    File size: 45.23 MB

================================================================================
TASK 1: GENE EXPRESSION BY BRAIN CELL CLUSTER TYPE
================================================================================
  Processing singlet clusters...
  Gene expression data prepared: 19,574 genes × 34 clusters

✓ Task 1 completed. Brain expression shape: (19574, 34)

================================================================================
TASK 2: EXTRACT DRUG TARGET GENES
================================================================================
  Processing DrugBank...
  Processing DRUGseqr...
  Total drugs identified: 8,427
  Drugs with gene targets: 3,245

✓ Task 2 completed. Drug targets: 3,245 drugs

... (Tasks 3-5 progress) ...

================================================================================
TASK 6: PREDICT & EVALUATE
================================================================================
  Accuracy: 0.8234
  Precision (weighted): 0.8156
  Recall (weighted): 0.8234
  F1-Score (weighted): 0.8167

✓ Task 6 completed.

================================================================================
✓ NESY SYSTEM PIPELINE COMPLETED SUCCESSFULLY
================================================================================

All results saved to: results
Check PROJECT_REPORT.md for detailed analysis
```

### Training with Custom Configuration

To modify pipeline configuration, edit the `config` dictionary in `nesy_system.py`:

```python
config = {
    'drugbank_xml': 'path/to/full_database.xml',      # DrugBank database
    'cluster_type_tsv': 'path/to/cluster_data.tsv',   # RNA cluster data
    'drugseqr_gmt': 'path/to/DRUGseqr.gmt',           # Gene set file
    'results_dir': 'results',                          # Output directory
    'top_k': 5                                         # Top-k predictions
}
```

Then run:
```bash
python nesy_system.py
```

### Prerequisites for Training

1. **Datasets** (download from [Google Drive](https://drive.google.com/drive/u/1/folders/1rnPldk5j06xEjuf3Tqk_dHZKW6ji0HOq?usp=sharing)):
   - `full_database.xml` - DrugBank complete database
   - `rna_single_nuclei_cluster_type.tsv` - scRNA gene expression data
   - `DRUGseqr.gmt` - Gene set file for drug targets

2. **Storage**: ~1-2 GB free disk space for intermediate results

3. **Memory**: 8GB+ RAM recommended (16GB+ for faster processing)

4. **Dependencies**: All required packages in `requirements.txt`

---

## Project Structure

```
├── interactive_predictor.py      # CLI interface
├── api_server.py                  # Web API backend
├── nesy_system.py                 # Full pipeline orchestrator
├── requirements.txt               # Python dependencies
├── docker-compose.yml             # Docker configuration
├── web/                           # Web interface (HTML/CSS/JS)
├── tasks/                         # Pipeline tasks (1-6)
│   ├── task1_brain_gene_expression.py
│   ├── task2_extract_drug_targets.py
│   ├── task3_fetch_atc_hierarchy.py
│   ├── task4_integrate_gene_signatures.py
│   ├── task5_train_classifier.py
│   ├── task5a_cns_classifier.py
│   └── task6_predict_evaluate.py
├── src/
│   ├── agents/
│   │   ├── sklearn_agent.py       # Model inference + drug similarity
│   │   └── evidence_agent.py      # Explanation generation
│   └── metrics/
│       └── brain_evidence.py      # BES/BSR calculation
├── dataset/                       # Raw data (DrugBank, RNA-seq)
├── models/                        # Trained models
└── results/                       # Outputs & analysis
```

---

## Features

✅ **Explainable Predictions** - Reasoning steps and evidence for each prediction  
✅ **Drug Similarity Matching** - Find drugs with overlapping target genes  
✅ **CNS Classification** - Distinguish CNS-active from non-CNS drugs  
✅ **Brain Evidence Metrics** - BES (Brain Expression Strength) and BSR (Brain Specificity Ratio)  
✅ **Multi-Interface** - CLI and web-based user interaction  
✅ **Containerized** - Docker support for easy deployment  

---

## Notes

- All intermediate results saved in `results/` directory
- Models trained on DrugBank + DRUGseqr datasets
- CNS classifier uses imbalanced data; predictions for rare classes less reliable
- Drug similarity search requires trained `task2_drug_targets.csv` file
