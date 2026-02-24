"""
Task 4: Integrate Gene Signatures
Input: Brain cell cluster type expression + Drug targets + Gene signatures
Output: Integrated feature embeddings
Process: Embedding (combine all information)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler

# Import BrainEvidenceMetrics for BES/BSR calculation
from src.metrics.brain_evidence import BrainEvidenceMetrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def create_drug_features(drug_genes: List[str], gene_expression: pd.DataFrame, bem: BrainEvidenceMetrics) -> np.ndarray:
    """
    Create feature vector for a drug based on its target genes' expression across brain cell cluster types, including BES and BSR.
    
    Args:
        drug_genes: List of gene names for the drug
        gene_expression: Gene expression matrix (genes × brain cell cluster types)
        bem: BrainEvidenceMetrics instance for BES/BSR calculation
    
    Returns:
        Feature vector representing drug's effect profile
    Note:
        The term 'brain_regions' is used for compatibility with BrainEvidenceMetrics, but refers to brain cell cluster types in this context.
    """
    available_genes = set(gene_expression.index)
    matching_genes = [g for g in drug_genes if g in available_genes]
    
    if not matching_genes:
        return None
    
    # Get expression profiles for matching genes
    gene_profiles = gene_expression.loc[matching_genes]
    
    # Create aggregated features
    features = []
    
    # 1. Mean expression per brain cell cluster type
    features.extend(gene_profiles.mean(axis=0).values)
    
    # 2. Max expression per brain cell cluster type
    features.extend(gene_profiles.max(axis=0).values)
    
    # 3. Standard deviation per brain cell cluster type
    features.extend(gene_profiles.std(axis=0).values)
    
    # 4. Overall statistics
    features.append(gene_profiles.values.mean())  # Overall mean
    features.append(gene_profiles.values.std())   # Overall std
    features.append(gene_profiles.values.max())   # Overall max
    features.append(len(matching_genes))          # Number of matching genes
    features.append(len(matching_genes) / len(drug_genes))  # Gene coverage ratio

    # 5. Compute BES and BSR using BrainEvidenceMetrics
    # Prepare gene_expression dict for BEM: {gene: {cell_type: expr, ...}, ...}
    gene_expr_dict = {g: {ct: float(gene_profiles.loc[g, ct]) for ct in gene_profiles.columns} for g in matching_genes}
    # Use the correct argument name 'brain_regions' for compatibility with BrainEvidenceMetrics (refers to cell cluster types)
    brain_regions = list(gene_profiles.columns)
    # For BSR, we need to define body_tissues (not present in this matrix, so use empty list)
    body_tissues = []
    bes = bem.brain_evidence_strength(gene_expr_dict, brain_regions)
    bsr = bem.brain_specificity_ratio(gene_expr_dict, brain_regions, body_tissues)
    features.append(bes)
    features.append(bsr)
    
    return np.array(features)


def execute(brain_expression: pd.DataFrame, 
            drug_targets: pd.DataFrame,
            output_dir: Path) -> Tuple[np.ndarray, List[int]]:
    """
    Execute Task 4: Integrate gene signatures into feature embeddings
    
    Args:
        brain_expression: Gene expression matrix from Task 1
        drug_targets: Drug-gene relationships from Task 2
        output_dir: Directory to save outputs
    
    Returns:
        Tuple of (feature_matrix, valid_indices)
    """
    print("\n[Task 4] Integrating gene signatures into feature embeddings...")
    
    print(f"\n  Input data:")
    print(f"    Brain genes: {brain_expression.shape[0]:,}")
    print(f"    Brain cell cluster types: {brain_expression.shape[1]}")
    print(f"    Drug entries: {len(drug_targets)}")
    
    # Filter to only DrugBank drugs (they have ATC codes)
    drugbank_drugs = drug_targets[drug_targets['source'] == 'DrugBank'].copy()
    print(f"\n  Filtering to DrugBank drugs with ATC codes: {len(drugbank_drugs)}")
    
    # Extract features for each drug
    features_list = []
    valid_indices = []
    failed_count = 0

    # Initialize BrainEvidenceMetrics
    bem = BrainEvidenceMetrics()

    print(f"\n  Extracting features for each drug...")
    for idx, row in tqdm(drugbank_drugs.iterrows(), total=len(drugbank_drugs), desc="    Processing drugs"):
        all_genes = row['all_genes']

        if not all_genes or not isinstance(all_genes, list):
            failed_count += 1
            continue

        # Create feature vector (now includes BES and BSR)
        features = create_drug_features(all_genes, brain_expression, bem)

        if features is not None:
            features_list.append(features)
            valid_indices.append(idx)
        else:
            failed_count += 1
    
    # Convert to numpy array
    feature_matrix = np.array(features_list)
    
    print(f"\n  Feature extraction results:")
    print(f"    Successful: {len(valid_indices)}")
    print(f"    Failed: {failed_count}")
    print(f"    Success rate: {len(valid_indices)/(len(valid_indices)+failed_count)*100:.1f}%")
    print(f"\n  Feature matrix shape: {feature_matrix.shape}")
    print(f"    Samples: {feature_matrix.shape[0]}")
    print(f"    Features per sample: {feature_matrix.shape[1]}")
    
    # Feature statistics
    print(f"\n  Feature statistics:")
    print(f"    Mean: {feature_matrix.mean():.4f}")
    print(f"    Std: {feature_matrix.std():.4f}")
    print(f"    Min: {feature_matrix.min():.4f}")
    print(f"    Max: {feature_matrix.max():.4f}")
    
    # Check for NaN or Inf values
    has_nan = np.isnan(feature_matrix).any()
    has_inf = np.isinf(feature_matrix).any()
    
    if has_nan or has_inf:
        print(f"\n  ⚠ Warning: Found NaN or Inf values, cleaning...")
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"    Cleaned feature matrix")
    
    # Standardize features
    print(f"\n  Standardizing features...")
    scaler = StandardScaler()
    feature_matrix_scaled = scaler.fit_transform(feature_matrix)
    
    print(f"    After standardization:")
    print(f"      Mean: {feature_matrix_scaled.mean():.4f}")
    print(f"      Std: {feature_matrix_scaled.std():.4f}")
    
    # Save scaler
    import joblib
    joblib.dump(scaler, output_dir / 'task4_feature_scaler.pkl')
    print(f"\n  Saved feature scaler to: task4_feature_scaler.pkl")
    
    # Save feature importance analysis
    feature_names = []
    n_cell_types = brain_expression.shape[1]
    cell_type_names = brain_expression.columns.tolist()

    for ct in cell_type_names:
        feature_names.append(f'mean_expr_{ct}')
    for ct in cell_type_names:
        feature_names.append(f'max_expr_{ct}')
    for ct in cell_type_names:
        feature_names.append(f'std_expr_{ct}')

    feature_names.extend(['overall_mean', 'overall_std', 'overall_max', 
                         'num_genes', 'gene_coverage', 'BES', 'BSR'])
    
    # Calculate feature variances
    feature_variances = feature_matrix_scaled.var(axis=0)
    feature_importance = pd.DataFrame({
        'feature_name': feature_names,
        'variance': feature_variances
    }).sort_values('variance', ascending=False)
    
    feature_importance.to_csv(output_dir / 'task4_feature_importance.csv', index=False)
    print(f"\n  Top 10 most variable features:")
    for _, row in feature_importance.head(10).iterrows():
        print(f"    {row['feature_name']}: {row['variance']:.4f}")
    

    # --- Plots for report ---
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 1. Feature variance (importance) bar plot
    plt.figure(figsize=(12, 6))
    feature_importance.head(20).plot(x='feature_name', y='variance', kind='bar', legend=False, color='dodgerblue')
    plt.title('Top 20 Most Variable Integrated Features')
    plt.ylabel('Variance (Feature Importance)')
    plt.xlabel('Feature')
    plt.xticks(rotation=75, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'task4_top20_feature_importance.png')
    plt.close()

    # 1b. Pie chart of variance contribution (top 10 features)
    plt.figure(figsize=(8, 8))
    top10 = feature_importance.head(10)
    plt.pie(top10['variance'], labels=top10['feature_name'], autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
    plt.title('Top 10 Feature Variance Contribution (Pie Chart)')
    plt.tight_layout()
    plt.savefig(output_dir / 'task4_top10_feature_variance_pie.png')
    plt.close()

    # 2. Heatmap of feature matrix (first 30 samples × first 30 features)
    plt.figure(figsize=(10, 8))
    sns.heatmap(feature_matrix_scaled[:30, :30], cmap='viridis')
    plt.title('Feature Matrix Heatmap (First 30 Samples × 30 Features)')
    plt.xlabel('Feature Index')
    plt.ylabel('Sample Index')
    plt.tight_layout()
    plt.savefig(output_dir / 'task4_feature_matrix_heatmap.png')
    plt.close()

    # 2b. Boxplot of all features (distribution across all samples)
    plt.figure(figsize=(14, 6))
    plt.boxplot(feature_matrix_scaled, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    plt.title('Boxplot: Distribution of All Features (Standardized)')
    plt.xlabel('Value (Standardized)')
    plt.tight_layout()
    plt.savefig(output_dir / 'task4_feature_boxplot.png')
    plt.close()

    # 3. Distribution of BES and BSR values
    plt.figure(figsize=(8, 5))
    bes_idx = feature_names.index('BES')
    bsr_idx = feature_names.index('BSR')
    plt.hist(feature_matrix_scaled[:, bes_idx], bins=40, color='purple', alpha=0.7, label='BES')
    plt.hist(feature_matrix_scaled[:, bsr_idx], bins=40, color='orange', alpha=0.7, label='BSR')
    plt.title('Distribution of BES and BSR (Standardized)')
    plt.xlabel('Value (Standardized)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'task4_bes_bsr_distribution.png')
    plt.close()

    # 3b. KDE plot for BES and BSR
    plt.figure(figsize=(8, 5))
    sns.kdeplot(feature_matrix_scaled[:, bes_idx], label='BES', fill=True, color='purple', alpha=0.5)
    sns.kdeplot(feature_matrix_scaled[:, bsr_idx], label='BSR', fill=True, color='orange', alpha=0.5)
    plt.title('KDE Plot of BES and BSR (Standardized)')
    plt.xlabel('Value (Standardized)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'task4_bes_bsr_kde.png')
    plt.close()

    # 4. Correlation heatmap of features (first 30 features)
    plt.figure(figsize=(12, 10))
    corr = pd.DataFrame(feature_matrix_scaled[:, :30]).corr()
    sns.heatmap(corr, cmap='coolwarm', center=0, annot=False)
    plt.title('Correlation Heatmap (First 30 Features)')
    plt.tight_layout()
    plt.savefig(output_dir / 'task4_feature_correlation_heatmap.png')
    plt.close()

    return feature_matrix_scaled, valid_indices


if __name__ == "__main__":
    # Test the task
    print("Task 4 requires outputs from Tasks 1 and 2")
    print("Run the full pipeline to test this task")
