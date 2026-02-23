"""
Script to create the union of all genes from task1_brain_expression.csv (index) and all_genes column in task2_drugbank_only.csv.
Saves the result as selected_genes.txt (one gene per line, uppercase, sorted).
"""

import pandas as pd
from pathlib import Path
import ast

# Paths
results_dir = Path("results")
expr_path = results_dir / "task1_brain_expression.csv"
drugbank_path = results_dir / "task2_drugbank_only.csv"
out_path = results_dir / "selected_genes.txt"

# Load genes from task1 (index)
expr_df = pd.read_csv(expr_path, index_col=0)
genes_task1 = set(expr_df.index.str.upper())

# Load genes from task2 (all_genes column)
drugbank_df = pd.read_csv(drugbank_path)
genes_task2 = set()
for genes_str in drugbank_df["all_genes"]:
    if pd.isna(genes_str):
        continue
    # Parse string representation of list
    try:
        genes = ast.literal_eval(genes_str)
        genes_task2.update(g.upper() for g in genes if isinstance(g, str))
    except Exception:
        continue

# Union and sort
genes_union = sorted(genes_task1.union(genes_task2))

# Save to file
with open(out_path, "w") as f:
    for gene in genes_union:
        f.write(gene + "\n")

print(f"Union gene list saved to {out_path} ({len(genes_union)} genes)")
