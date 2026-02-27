import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict


def execute(cluster_type_path: str, output_dir: Path) -> pd.DataFrame:
    """
    Execute Task 1: Load and process brain gene expression data
    
    Args:
        cluster_type_path: Path to rna_single_nuclei_cluster_type.tsv
        output_dir: Directory to save outputs
    
    Returns:
        DataFrame with gene expression matrix (genes × cluster types)
    """
    print("\n[Task 1] Loading brain gene expression data...")
    
    # Load the cluster type data
    df = pd.read_csv(cluster_type_path, sep='\t')
    
    print(f"  Loaded {len(df):,} gene-cluster pairs")
    print(f"  Unique genes: {df['Gene name'].nunique():,}")
    print(f"  Unique cluster types: {df['Cluster type'].nunique()}")
    
    # Show cluster type distribution
    print(f"\n  Brain region cluster types:")
    for cluster_type in sorted(df['Cluster type'].unique()):
        count = (df['Cluster type'] == cluster_type).sum()
        print(f"    - {cluster_type}: {count:,} genes")
    
    # Create gene expression matrix: genes × cluster types
    print("\n  Creating gene expression matrix...")
    gene_expression = df.pivot_table(
        index='Gene name',
        columns='Cluster type',
        values='nCPM',
        aggfunc='mean'
    ).fillna(0)
    
    print(f"  Matrix shape: {gene_expression.shape}")
    print(f"  Genes: {gene_expression.shape[0]:,}")
    print(f"  Brain regions (cluster types): {gene_expression.shape[1]}")
    
    # Calculate statistics
    print("\n  Expression statistics:")
    print(f"    Mean expression: {gene_expression.values.mean():.2f}")
    print(f"    Median expression: {np.median(gene_expression.values):.2f}")
    print(f"    Max expression: {gene_expression.values.max():.2f}")
    print(f"    Non-zero values: {(gene_expression.values > 0).sum() / gene_expression.size * 100:.1f}%")
    
    # Identify highly expressed genes across regions
    mean_expression_per_gene = gene_expression.mean(axis=1)
    top_genes = mean_expression_per_gene.nlargest(10)
    print("\n  Top 10 highly expressed genes across all brain regions:")
    for gene, expr in top_genes.items():
        print(f"    {gene}: {expr:.2f}")

        # Create a heatmap of gene expression (genes × cluster types)
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(16, 8))
        # For visualization, limit to top 30 genes by mean expression
        top_gene_names = mean_expression_per_gene.nlargest(15).index
        heatmap_data = gene_expression.loc[top_gene_names]
        sns.heatmap(heatmap_data, cmap="viridis", cbar_kws={"label": "Expression (nCPM)"}, linewidths=0.5)
        plt.title('Gene Expression Heatmap (Top 15 Genes × Brain Cluster Types)')
        plt.xlabel('Brain Cluster Type')
        plt.ylabel('Gene Name')
        plt.tight_layout()
        plot_path = output_dir / 'task1_expression_heatmap.png'
        plt.savefig(plot_path)
        plt.close()
        print(f"\n[OK] Heatmap saved: {plot_path}")
    
    return gene_expression


if __name__ == "__main__":
    # Test the task
    cluster_type_path = r"c:\Users\user\OneDrive\Desktop\My Files\LUH\Semester 5\DrugCategoryPrediction\dataset\rna_single_nuclei_cluster_type.tsv\rna_single_nuclei_cluster_type.tsv"
    output_dir = Path("../results")
    output_dir.mkdir(exist_ok=True)
    
    result = execute(cluster_type_path, output_dir)
    print(f"\n[OK] Task 1 test completed. Shape: {result.shape}")
