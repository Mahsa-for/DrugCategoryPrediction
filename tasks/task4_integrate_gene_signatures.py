"""
Task 4: Integrate Gene Signatures
Input: Brain expression + Drug targets + Gene signatures
Output: Integrated feature embeddings
Process: Embedding (combine all information)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler


def create_drug_features(drug_genes: List[str], gene_expression: pd.DataFrame) -> np.ndarray:
    """
    Create feature vector for a drug based on its target genes' expression
    
    Args:
        drug_genes: List of gene names for the drug
        gene_expression: Gene expression matrix (genes × brain regions)
    
    Returns:
        Feature vector representing drug's effect profile
    """
    available_genes = set(gene_expression.index)
    matching_genes = [g for g in drug_genes if g in available_genes]
    
    if not matching_genes:
        return None
    
    # Get expression profiles for matching genes
    gene_profiles = gene_expression.loc[matching_genes]
    
    # Create aggregated features
    features = []
    
    # 1. Mean expression per brain region
    features.extend(gene_profiles.mean(axis=0).values)
    
    # 2. Max expression per brain region
    features.extend(gene_profiles.max(axis=0).values)
    
    # 3. Standard deviation per brain region
    features.extend(gene_profiles.std(axis=0).values)
    
    # 4. Overall statistics
    features.append(gene_profiles.values.mean())  # Overall mean
    features.append(gene_profiles.values.std())   # Overall std
    features.append(gene_profiles.values.max())   # Overall max
    features.append(len(matching_genes))          # Number of matching genes
    features.append(len(matching_genes) / len(drug_genes))  # Gene coverage ratio
    
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
    print(f"    Brain regions: {brain_expression.shape[1]}")
    print(f"    Drug entries: {len(drug_targets)}")
    
    # Filter to only DrugBank drugs (they have ATC codes)
    drugbank_drugs = drug_targets[drug_targets['source'] == 'DrugBank'].copy()
    print(f"\n  Filtering to DrugBank drugs with ATC codes: {len(drugbank_drugs)}")
    
    # Extract features for each drug
    features_list = []
    valid_indices = []
    failed_count = 0
    
    print(f"\n  Extracting features for each drug...")
    for idx, row in tqdm(drugbank_drugs.iterrows(), total=len(drugbank_drugs), desc="    Processing drugs"):
        all_genes = row['all_genes']
        
        if not all_genes or not isinstance(all_genes, list):
            failed_count += 1
            continue
        
        # Create feature vector
        features = create_drug_features(all_genes, brain_expression)
        
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
    n_regions = brain_expression.shape[1]
    region_names = brain_expression.columns.tolist()
    
    for region in region_names:
        feature_names.append(f'mean_expr_{region}')
    for region in region_names:
        feature_names.append(f'max_expr_{region}')
    for region in region_names:
        feature_names.append(f'std_expr_{region}')
    
    feature_names.extend(['overall_mean', 'overall_std', 'overall_max', 
                         'num_genes', 'gene_coverage'])
    
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
    
    return feature_matrix_scaled, valid_indices


if __name__ == "__main__":
    # Test the task
    print("Task 4 requires outputs from Tasks 1 and 2")
    print("Run the full pipeline to test this task")
