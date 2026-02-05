"""
Visualization and Analysis of Drug Category Prediction Results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def analyze_drug_data():
    """Analyze the parsed drug data"""
    print("=" * 60)
    print("DRUG DATA ANALYSIS")
    print("=" * 60)
    
    # Load drug data
    drugs_df = pd.read_csv('results/drugs_data.csv')
    
    print(f"\nTotal drugs parsed: {len(drugs_df)}")
    print(f"\nSample drugs:")
    print(drugs_df[['name', 'num_genes']].head(10).to_string(index=False))
    
    # Analyze gene counts
    print(f"\n### Gene Statistics ###")
    print(f"Average genes per drug: {drugs_df['num_genes'].mean():.2f}")
    print(f"Median genes per drug: {drugs_df['num_genes'].median():.0f}")
    print(f"Max genes per drug: {drugs_df['num_genes'].max():.0f}")
    print(f"Min genes per drug: {drugs_df['num_genes'].min():.0f}")
    
    # Plot gene distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Gene count distribution
    axes[0, 0].hist(drugs_df['num_genes'], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Number of Target Genes')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Target Genes per Drug')
    axes[0, 0].axvline(drugs_df['num_genes'].mean(), color='red', 
                       linestyle='--', label=f'Mean: {drugs_df["num_genes"].mean():.1f}')
    axes[0, 0].legend()
    
    # 2. ATC category distribution
    all_atc = []
    for atc_list in drugs_df['atc_codes'].apply(eval):
        all_atc.extend([code[0] for code in atc_list])
    
    atc_counts = Counter(all_atc)
    atc_df = pd.DataFrame(list(atc_counts.items()), columns=['Category', 'Count'])
    atc_df = atc_df.sort_values('Count', ascending=False)
    
    axes[0, 1].bar(atc_df['Category'], atc_df['Count'], edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('ATC Category')
    axes[0, 1].set_ylabel('Number of Drugs')
    axes[0, 1].set_title('Drug Distribution Across ATC Categories')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Gene type distribution
    target_total = drugs_df['target_genes'].apply(lambda x: len(eval(x))).sum()
    enzyme_total = drugs_df['enzyme_genes'].apply(lambda x: len(eval(x))).sum()
    transporter_total = drugs_df['transporter_genes'].apply(lambda x: len(eval(x))).sum()
    
    gene_types = ['Targets', 'Enzymes', 'Transporters']
    gene_counts = [target_total, enzyme_total, transporter_total]
    
    axes[1, 0].pie(gene_counts, labels=gene_types, autopct='%1.1f%%', startangle=90)
    axes[1, 0].set_title('Gene Type Distribution')
    
    # 4. Drugs per gene count
    gene_count_dist = drugs_df['num_genes'].value_counts().sort_index()
    gene_count_dist_top = gene_count_dist.head(20)
    
    axes[1, 1].bar(gene_count_dist_top.index, gene_count_dist_top.values, 
                   edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Number of Genes')
    axes[1, 1].set_ylabel('Number of Drugs')
    axes[1, 1].set_title('Drugs by Gene Count (Top 20)')
    
    plt.tight_layout()
    plt.savefig('results/drug_data_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nSaved drug data analysis to: results/drug_data_analysis.png")


def analyze_gene_expression():
    """Analyze brain gene expression data"""
    print("\n" + "=" * 60)
    print("GENE EXPRESSION ANALYSIS")
    print("=" * 60)
    
    # Load gene expression
    gene_expr = pd.read_csv('results/gene_expression.csv', index_col=0)
    
    print(f"\nGene expression matrix shape: {gene_expr.shape}")
    print(f"Number of genes: {gene_expr.shape[0]}")
    print(f"Number of brain cell types: {gene_expr.shape[1]}")
    
    print(f"\n### Expression Statistics ###")
    print(f"Mean expression: {gene_expr.values.mean():.2f}")
    print(f"Median expression: {np.median(gene_expr.values):.2f}")
    print(f"Max expression: {gene_expr.values.max():.2f}")
    
    print(f"\n### Brain Cell Types ###")
    for i, cell_type in enumerate(gene_expr.columns[:10], 1):
        print(f"{i}. {cell_type}")
    print(f"... and {len(gene_expr.columns) - 10} more")
    
    # Create heatmap of top variable genes
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. Top variable genes across cell types
    gene_std = gene_expr.std(axis=1)
    top_genes = gene_std.nlargest(30).index
    top_gene_expr = gene_expr.loc[top_genes]
    
    sns.heatmap(top_gene_expr, cmap='viridis', ax=axes[0], 
                cbar_kws={'label': 'nCPM'}, yticklabels=True)
    axes[0].set_title('Top 30 Variable Genes Across Brain Cell Types')
    axes[0].set_xlabel('Cell Type')
    axes[0].set_ylabel('Gene')
    
    # 2. Average expression per cell type
    cell_type_means = gene_expr.mean(axis=0).sort_values(ascending=False)
    axes[1].barh(range(len(cell_type_means)), cell_type_means.values, alpha=0.7)
    axes[1].set_yticks(range(len(cell_type_means)))
    axes[1].set_yticklabels(cell_type_means.index, fontsize=8)
    axes[1].set_xlabel('Average Expression (nCPM)')
    axes[1].set_title('Average Gene Expression by Cell Type')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('results/gene_expression_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nSaved gene expression analysis to: results/gene_expression_analysis.png")


def create_summary_report():
    """Create a comprehensive summary report"""
    print("\n" + "=" * 60)
    print("GENERATING COMPREHENSIVE SUMMARY REPORT")
    print("=" * 60)
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Load data
    drugs_df = pd.read_csv('results/drugs_data.csv')
    
    # 1. Pipeline workflow
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')
    workflow_text = """
    DRUG CATEGORY PREDICTION PIPELINE
    
    1. DrugBank XML → Parse Drugs (2077 drugs with targets & ATC codes)
    2. Brain scRNA-seq → Gene Expression Matrix (19574 genes × 34 cell types)
    3. Feature Engineering → Drug-Gene Expression Vectors (1983 drugs × 136 features)
    4. Stage 1: CNS vs Non-CNS Classification (87.4% accuracy)
    5. Stage 2: Drug Category Prediction (45.1% accuracy, 14 classes)
    """
    ax1.text(0.1, 0.5, workflow_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', 
             facecolor='wheat', alpha=0.5))
    ax1.set_title('Pipeline Overview', fontsize=14, fontweight='bold', pad=20)
    
    # 2. Model performance comparison
    ax2 = fig.add_subplot(gs[1, 0])
    models = ['Random Forest', 'Gradient\nBoosting', 'SVM', 'Logistic\nRegression']
    cns_scores = [0.8796, 0.8796, 0.8657, 0.8644]
    category_scores = [0.4388, 0.4180, 0.2692, 0.3241]
    
    x = np.arange(len(models))
    width = 0.35
    ax2.bar(x - width/2, cns_scores, width, label='CNS Classifier', alpha=0.8)
    ax2.bar(x + width/2, category_scores, width, label='Category Classifier', alpha=0.8)
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Model Performance Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=9)
    ax2.legend()
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. CNS vs Non-CNS results
    ax3 = fig.add_subplot(gs[1, 1])
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [0.8741, 0.7500, 0.4286, 0.5455]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    ax3.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Score')
    ax3.set_title('CNS vs Non-CNS Classification Metrics')
    ax3.set_ylim(0, 1.0)
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Category prediction results
    ax4 = fig.add_subplot(gs[1, 2])
    metrics_cat = ['Accuracy', 'F1-Macro', 'F1-Weighted']
    values_cat = [0.4509, 0.3364, 0.4299]
    colors_cat = ['#9b59b6', '#1abc9c', '#e67e22']
    ax4.bar(metrics_cat, values_cat, color=colors_cat, alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Score')
    ax4.set_title('Drug Category Prediction Metrics')
    ax4.set_ylim(0, 1.0)
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. ATC category distribution
    ax5 = fig.add_subplot(gs[2, 0])
    all_atc = []
    for atc_list in drugs_df['atc_codes'].apply(eval):
        all_atc.extend([code[0] for code in atc_list])
    atc_counts = Counter(all_atc)
    atc_df = pd.DataFrame(list(atc_counts.items()), columns=['Category', 'Count'])
    atc_df = atc_df.sort_values('Count', ascending=False).head(10)
    ax5.barh(atc_df['Category'], atc_df['Count'], alpha=0.7, edgecolor='black')
    ax5.set_xlabel('Number of Drugs')
    ax5.set_title('Top 10 Drug Categories')
    ax5.invert_yaxis()
    
    # 6. Gene count distribution
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.hist(drugs_df['num_genes'], bins=25, edgecolor='black', alpha=0.7, color='steelblue')
    ax6.set_xlabel('Number of Target Genes')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Distribution of Target Genes per Drug')
    ax6.axvline(drugs_df['num_genes'].mean(), color='red', 
                linestyle='--', linewidth=2, label=f'Mean: {drugs_df["num_genes"].mean():.1f}')
    ax6.legend()
    
    # 7. Summary statistics
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    summary_stats = f"""
    SUMMARY STATISTICS
    
    Drugs Processed: {len(drugs_df)}
    Valid Features: 1983
    
    CNS Drugs: 349 (17.6%)
    Non-CNS Drugs: 1634 (82.4%)
    
    ATC Categories: 14
    Brain Cell Types: 34
    Total Genes: 19,574
    
    Best CNS Model:
      Random Forest (87.96%)
    
    Best Category Model:
      Random Forest (43.88%)
    """
    ax7.text(0.1, 0.5, summary_stats, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', 
             facecolor='lightblue', alpha=0.5))
    
    plt.suptitle('Drug Category Prediction - Comprehensive Summary', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig('results/comprehensive_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nSaved comprehensive summary to: results/comprehensive_summary.png")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RESULTS ANALYSIS AND VISUALIZATION")
    print("=" * 60)
    
    analyze_drug_data()
    analyze_gene_expression()
    create_summary_report()
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print("\nGenerated visualizations:")
    print("  1. results/drug_data_analysis.png")
    print("  2. results/gene_expression_analysis.png")
    print("  3. results/comprehensive_summary.png")
    print("  4. results/confusion_matrix_cns.png")
    print("  5. results/confusion_matrix_category.png")
    print("  6. results/model_comparison_cns.png")
    print("  7. results/model_comparison_category.png")
    print("\nResults summary: results/results_summary.txt")
