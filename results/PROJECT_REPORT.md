# Drug Category Prediction Based on Gene Expression in Brain

## Student Project Report

---

## 1. Introduction

This project implements a Neuro-Symbolic (NeSy) system for predicting drug therapeutic categories based on their effects on gene expression in brain cell cluster types.

## 2. System Architecture

The system follows a modular design pattern with 6 main tasks:

### Task 1: Gene Expression by Brain Cell Cluster Type
- **Description**: Analyze scRNA-seq data to understand gene expression patterns across different brain cell cluster types
- **Output File**: `task1_*.csv` or `task1_*.npy`

### Task 2: Extract Drug Target Genes
- **Description**: Combine DrugBank and DRUGseqr.gmt to extract comprehensive drug-gene interactions
- **Output File**: `task2_*.csv` or `task2_*.npy`

### Task 3: Fetch ATC Hierarchy
- **Description**: Extract therapeutic classification from DrugBank ATC codes
- **Output File**: `task3_*.csv` or `task3_*.npy`

### Task 4: Integrate Gene Signatures
- **Description**: Create feature embeddings combining brain cell cluster type expression, drug targets, and gene signatures
- **Output File**: `task4_*.csv` or `task4_*.npy`

### Task 5: CNS vs Non-CNS Classification (Stage 1)
- **Description**: Train binary classifier to distinguish CNS-active from Non-CNS drugs
- **Output File**: `task5_*.csv` or `task5_*.npy`

### Task 6: Drug Category Classification (Stage 2)
- **Description**: Learn patterns to predict drug therapeutic categories based on gene effects
- **Output File**: `task6_*.csv` or `task6_*.npy`

### Task 7: Prediction and Evaluation
- **Description**: Generate top-k predictions with probabilities and comprehensive evaluation metrics
- **Output File**: `task7_*.csv` or `task7_*.npy`

## 3. Datasets Used

1. **DrugBank XML**: Comprehensive drug database with targets, ATC codes
2. **DRUGseqr.gmt**: Gene expression signatures from drug perturbations
3. **scRNA Brain Data**: Single-cell RNA-seq from brain cell cluster types

## 4. Results and Evaluation

### CNS Classification (Stage 1) Performance

- **best_model**: Gradient Boosting
- **cv_accuracy**: 0.8941
- **cv_std**: 0.0074
- **train_accuracy**: 0.9491
- **train_f1**: 0.8237
- **train_precision**: 0.9593
- **train_recall**: 0.7217
- **num_cns_samples**: 327.0000
- **num_non_cns_samples**: 1656.0000
- **class_distribution**: {'CNS': 327, 'Non-CNS': 1656}

### Drug Category Classification (Stage 2) Performance

- **best_model**: Random Forest
- **cv_accuracy**: 0.46215503045453643
- **train_accuracy**: 0.9394703656998739
- **num_classes**: 14
- **class_names**: ['A', 'B', 'C', 'D', 'G', 'H', 'J', 'L', 'M', 'N', 'P', 'R', 'S', 'V']

### Test Performance

- **accuracy**: 0.4509
- **precision_macro**: 0.3513
- **precision_weighted**: 0.4274
- **recall_macro**: 0.3255
- **recall_weighted**: 0.4509
- **f1_macro**: 0.3263
- **f1_weighted**: 0.4267
- **per_class_report**: {'A': {'precision': 0.3404255319148936, 'recall': 0.36363636363636365, 'f1-score': 0.3516483516483517, 'support': 44.0}, 'B': {'precision': 0.42857142857142855, 'recall': 0.2857142857142857, 'f1-score': 0.34285714285714286, 'support': 21.0}, 'C': {'precision': 0.46, 'recall': 0.48936170212765956, 'f1-score': 0.4742268041237113, 'support': 47.0}, 'D': {'precision': 0.5454545454545454, 'recall': 0.3157894736842105, 'f1-score': 0.4, 'support': 19.0}, 'G': {'precision': 0.375, 'recall': 0.36, 'f1-score': 0.3673469387755102, 'support': 25.0}, 'H': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 9.0}, 'J': {'precision': 0.40816326530612246, 'recall': 0.5555555555555556, 'f1-score': 0.47058823529411764, 'support': 36.0}, 'L': {'precision': 0.4166666666666667, 'recall': 0.6730769230769231, 'f1-score': 0.5147058823529411, 'support': 52.0}, 'M': {'precision': 0.36363636363636365, 'recall': 0.2, 'f1-score': 0.25806451612903225, 'support': 20.0}, 'N': {'precision': 0.6153846153846154, 'recall': 0.7384615384615385, 'f1-score': 0.6713286713286714, 'support': 65.0}, 'P': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 8.0}, 'R': {'precision': 0.6923076923076923, 'recall': 0.375, 'f1-score': 0.4864864864864865, 'support': 24.0}, 'S': {'precision': 0.2727272727272727, 'recall': 0.2, 'f1-score': 0.23076923076923078, 'support': 15.0}, 'V': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 12.0}, 'accuracy': 0.4508816120906801, 'macro avg': {'precision': 0.35130981299782865, 'recall': 0.32547113158975266, 'f1-score': 0.32628730426894254, 'support': 397.0}, 'weighted avg': {'precision': 0.4273977958806039, 'recall': 0.4508816120906801, 'f1-score': 0.42666400802223997, 'support': 397.0}}
- **confusion_matrix**: [[16  0  6  0  3  0  1  8  1  8  0  1  0  0]
 [ 2  6  1  0  2  0  3  5  0  2  0  0  0  0]
 [ 1  2 23  0  1  0  5  5  1  7  0  0  1  1]
 [ 0  1  3  6  1  0  4  4  0  0  0  0  0  0]
 [ 2  0  2  0  9  1  4  4  0  0  0  0  3  0]
 [ 3  0  0  3  0  0  0  2  1  0  0  0  0  0]
 [ 1  0  3  0  1  0 20  6  0  1  1  1  2  0]
 [ 5  3  2  0  2  0  2 35  0  2  1  0  0  0]
 [ 4  0  0  0  0  0  2  3  4  5  0  1  0  1]
 [ 6  0  2  1  2  0  1  3  1 48  0  0  1  0]
 [ 1  0  1  0  0  0  3  2  0  1  0  0  0  0]
 [ 1  1  2  1  1  0  3  2  1  2  0  9  1  0]
 [ 3  0  1  0  2  0  0  2  2  1  0  1  3  0]
 [ 2  1  4  0  0  0  1  3  0  1  0  0  0  0]]
- **class_names**: ['A', 'B', 'C', 'D', 'G', 'H', 'J', 'L', 'M', 'N', 'P', 'R', 'S', 'V']
- **top_2_accuracy**: 0.5894
- **top_3_accuracy**: 0.6599
- **top_5_accuracy**: 0.7607

## 5. Explainability

Each task is independently implemented and can be analyzed:
- Task outputs are saved as CSV/NPY files for inspection
- Feature importance can be extracted from the classification model
- Top-k predictions provide confidence scores for interpretability

## 6. Conclusion

This NeSy system successfully integrates multiple data sources to predict drug therapeutic categories based on gene expression effects in brain cell cluster types. The modular architecture ensures each step is explainable and verifiable.

## 7. Generated Files

```
results/
  - task1_brain_expression.csv
  - task1_expression_heatmap.png
  - task2_drug_targets.csv
  - task2_drugbank_only.csv
  - task2_drugseqr_only.csv
  - task2_gene_type_proportion.png
  - task2_top20_variable_genes.png
  - task3_atc_codes_per_drug.png
  - task3_atc_codes_per_drug_histogram.png
  - task3_atc_hierarchy.csv
  - task3_atc_level1_distribution.png
  - task3_atc_level_statistics.csv
  - task3_unique_categories_per_level.png
  - task4_bes_bsr_scatter.png
  - task4_feature_correlation_heatmap.png
  - task4_feature_importance.csv
  - task4_feature_matrix_pca.png
  - task4_feature_scaler.pkl
  - task4_integrated_features.npy
  - task4_valid_indices.npy
  - task5_classifier_model.pkl
  - task5_label_encoder.pkl
  - task5a_class_distribution_pie.png
  - task5a_cns_classifier.pkl
  - task5a_cns_score_calibrator.pkl
  - task5a_confusion_matrix_threshold_0.3.png
  - task5a_model_comparison.csv
  - task5a_model_comparison_bar.png
  - task5a_model_comparison_horizontal.png
  - task5a_model_comparison_lollipop.png
  - task5a_roc_curve.png
  - task6_classification_report.csv
  - task6_confidence_boxplot.png
  - task6_confusion_matrix.png
  - task6_example_predictions.csv
  - task6_per_class_f1_heatmap.png
  - task6_per_class_metrics.png
  - task6_prediction_outcome_counts.png
  - task6_predictions.csv
  - task6_test_class_distribution_pie.png
  - task6_topk_accuracy.png
  - PROJECT_REPORT.md
  - EVALUATION_METRICS.json
```
