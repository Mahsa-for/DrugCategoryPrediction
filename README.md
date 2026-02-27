# DrugCategoryPrediction

# Drug Category Prediction System

This project implements a Neuro-Symbolic (NeSy) system for predicting drug therapeutic categories based on gene expression and drug target information. The system is modular, supporting both command-line and web-based user interaction.

## System Overview

The pipeline consists of several stages:
1. **Data Validation**
2. **Gene Expression Processing**
3. **Drug Target Extraction**
4. **ATC Hierarchy Extraction**
5. **Feature Integration**
6. **Model Training (CNS and Category Classification)**
7. **Prediction and Reasoning**
8. **User Interaction (CLI & Web API)**

---

## File/Module Responsibilities

### Top-Level Scripts
- **nesy_system.py**: Main orchestrator for the modular pipeline. Runs each task in sequence and manages configuration.
- **drug_prediction_pipeline.py**: Implements the core pipeline logic for data validation, feature extraction, model training, and evaluation. Can be run standalone for full pipeline execution.
- **interactive_predictor.py**: Provides an interactive command-line interface for users to input genes and receive predictions and explanations.
- **api_server.py**: Flask backend API for serving predictions and explanations to the web interface. Uses `interactive_predictor.py` for backend logic.

### Task Modules (`tasks/`)
- **task1_brain_gene_expression.py**: Processes single-cell RNA-seq data to generate gene expression matrices for brain cell cluster types.
- **task2_extract_drug_targets.py**: Extracts drug target, enzyme, and transporter genes from DrugBank and DRUGseqr datasets.
- **task3_fetch_atc_hierarchy.py**: Extracts ATC (Anatomical Therapeutic Chemical) hierarchy and categories for drugs.
- **task4_integrate_gene_signatures.py**: Integrates gene expression and drug target data into feature vectors for each drug.
- **task5b_train_classifier.py*_classifier.py**: Trains machine learning models for drug category classification.
- **task5a_cns_classifier.py**: Trains a classifier to distinguish CNS-active drugs from non-CNS drugs.
- **task6_predict_evaluate.py**: Runs predictions on test data and evaluates model performance.
- **create_selected_genes_union.py**: (Optional) Generates a union list of all genes used in the system for validation or UI purposes.

### Supporting Modules (`src/`)
- **src/agents/evidence_agent.py**: Provides reasoning and explanation for predictions, integrating evidence from multiple sources.
- **src/agents/sklearn_agent.py**: Supports model-based reasoning and explanation using scikit-learn models. Also implements drug similarity search by target gene overlap using the `results/task2_drug_targets.csv` file.
- **src/metrics/brain_evidence.py**: Calculates brain evidence metrics (e.g., BES, BSR) for gene sets.

### Web Interface
- **web/index.html**: Frontend for user interaction via browser.
- **package.json, vite.config.js**: Frontend build and configuration files.

### Data and Results
- **dataset/**: Contains raw input datasets (DrugBank XML, gene expression, etc.).
- **models/**: Stores trained model files.
- **results/**: Stores all intermediate and final outputs (CSV, NPY, reports, plots).
	- **task2_drug_targets.csv**: Contains drug target gene lists for each drug. Used for drug similarity search by gene overlap in the web and CLI interfaces.

---

## Workflow

1. **Validation**: The system checks the existence and integrity of required datasets.
2. **Gene Expression Processing**: Generates a matrix of gene expression across brain cell cluster types.
3. **Drug Target Extraction**: Extracts all relevant genes for each drug from DrugBank and DRUGseqr.
4. **ATC Hierarchy Extraction**: Assigns ATC categories to drugs.
5. **Feature Integration**: Combines gene expression and drug target data into feature vectors.
6. **Model Training**:
    - **CNS Classifier**: Trains a model to classify drugs as CNS-active or not.
    - **Category Classifier**: Trains a model to predict the therapeutic category of each drug.
7. **Prediction & Reasoning**: Predicts categories for new gene sets and provides explanations using evidence agents.
8. **Drug Similarity Search**: Finds drugs with overlapping target genes using the `task2_drug_targets.csv` file. Results are shown in both CLI and web interface.
9. **User Interaction**:
    - **CLI**: Via `interactive_predictor.py` for direct user input.
    - **Web API**: Via `api_server.py` and `web/index.html` for browser-based interaction.

---

## How to Use

- **Full Pipeline**: Run `python drug_prediction_pipeline.py` or use `nesy_system.py` for modular execution.
- **Interactive CLI**: Run `python interactive_predictor.py`.
- **Web Interface**: Start the API server with `python api_server.py` and open the web interface.

---

## Notes
- All intermediate and final results are saved in the `results/` directory.
- The system is modular; each task can be run and debugged independently.
- Drug similarity search is available in both CLI and web interface, using gene overlap from `task2_drug_targets.csv`.
- For detailed explanations and reasoning, the system uses evidence agents and brain evidence metrics (BES, BSR) based on brain cell cluster types.
