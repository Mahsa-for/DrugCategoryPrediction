from pathlib import Path
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
from typing import Dict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def generate_task2_plots(combined_df: pd.DataFrame, output_dir: Path):
    """
    Generate only the two required plots for Task 2.
    """
    sns.set(style="whitegrid", context="paper", font_scale=1.2)

    # 1. Pie chart: Proportion of Gene Types (DrugBank only)
    drugbank_only = combined_df[combined_df['source'] == 'DrugBank']
    total_targets = sum(len(t) for t in drugbank_only['target_genes'])
    total_enzymes = sum(len(e) for e in drugbank_only['enzyme_genes'])
    total_transporters = sum(len(t) for t in drugbank_only['transporter_genes'])
    gene_type_counts = pd.Series({
        "Target Genes": total_targets,
        "Enzyme Genes": total_enzymes,
        "Transporter Genes": total_transporters
    })
    plt.figure(figsize=(6, 6))
    gene_type_counts.plot(
        kind='pie',
        autopct='%1.1f%%',
        colors=['#4C72B0', '#DD8452', '#55A868'],
        startangle=90
    )
    plt.ylabel("")
    plt.title("Proportion of Gene Types (DrugBank)")
    plt.tight_layout()
    plt.savefig(output_dir / "task2_gene_type_proportion.png", dpi=300)
    plt.close()

    # 2. Barplot: Top 20 Most Variable Genes Across Drugs
    all_genes_set = set()
    for genes in combined_df['all_genes']:
        if isinstance(genes, list):
            all_genes_set.update(genes)
    all_genes = sorted(list(all_genes_set))
    drug_gene_matrix = pd.DataFrame(0, index=combined_df.index, columns=all_genes)
    for idx, row in combined_df.iterrows():
        if isinstance(row['all_genes'], list):
            for gene in row['all_genes']:
                if gene in drug_gene_matrix.columns:
                    drug_gene_matrix.at[idx, gene] = 1
    gene_variances = drug_gene_matrix.var(axis=0)
    top_genes = gene_variances.nlargest(20)
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=top_genes.values,
        y=top_genes.index,
        palette="viridis"
    )
    plt.xlabel("Variance Across Drugs")
    plt.ylabel("Gene Symbol")
    plt.title("Top 20 Most Variable Genes Across Drugs")
    plt.tight_layout()
    plt.savefig(output_dir / "task2_top20_variable_genes.png", dpi=300)
    plt.close()

def parse_drugseqr_gmt(gmt_path: str) -> Dict[str, Dict]:
    """
    Parse DRUGseqr.gmt file.
    """
    print("\n  Parsing DRUGseqr.gmt...")
    drugs_data = {}
    with open(gmt_path, 'r') as f:
        for line_num, line in enumerate(tqdm(f, desc="    Reading DRUGseqr"), 1):
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            drug_name = parts[0]
            drug_info = parts[1]
            genes = parts[2:]
            info_parts = drug_info.split(';')
            cid = info_parts[0].replace('CID:', '') if len(info_parts) > 0 else ''
            direction = 'unknown'
            if ' up' in drug_name.lower():
                direction = 'up'
            elif ' down' in drug_name.lower():
                direction = 'down'
            clean_name = drug_name.split('_')[0] if '_' in drug_name else drug_name
            drugs_data[drug_name] = {
                'drug_name': clean_name,
                'cid': cid,
                'direction': direction,
                'genes': genes,
                'num_genes': len(genes)
            }
    print(f"    Parsed {len(drugs_data)} drug signatures")
    return drugs_data

def parse_drugbank_xml(xml_path: str) -> pd.DataFrame:
    """
    Parse DrugBank XML to extract drug-gene interactions.
    """
    print("\n  Parsing DrugBank XML...")
    namespace = {'db': 'http://www.drugbank.ca'}
    drugs_data = []
    context = ET.iterparse(xml_path, events=('start', 'end'))
    context = iter(context)
    event, root = next(context)
    for event, elem in tqdm(context, desc="    Parsing DrugBank"):
        if event == 'end' and elem.tag == '{http://www.drugbank.ca}drug':
            drug_id_elem = elem.find('db:drugbank-id[@primary="true"]', namespace)
            name_elem = elem.find('db:name', namespace)
            if drug_id_elem is None or name_elem is None:
                elem.clear()
                root.clear()
                continue
            drug_id = drug_id_elem.text
            name = name_elem.text
            atc_codes = [atc.get('code') for atc in elem.findall('db:atc-codes/db:atc-code', namespace) if atc.get('code')]
            target_genes = []
            for target in elem.findall('db:targets/db:target', namespace):
                for poly in target.findall('db:polypeptide', namespace):
                    gene_name_elem = poly.find('db:gene-name', namespace)
                    if gene_name_elem is not None and gene_name_elem.text:
                        target_genes.append(gene_name_elem.text)
            enzyme_genes = []
            for enzyme in elem.findall('db:enzymes/db:enzyme', namespace):
                for poly in enzyme.findall('db:polypeptide', namespace):
                    gene_name_elem = poly.find('db:gene-name', namespace)
                    if gene_name_elem is not None and gene_name_elem.text:
                        enzyme_genes.append(gene_name_elem.text)
            transporter_genes = []
            for transporter in elem.findall('db:transporters/db:transporter', namespace):
                for poly in transporter.findall('db:polypeptide', namespace):
                    gene_name_elem = poly.find('db:gene-name', namespace)
                    if gene_name_elem is not None and gene_name_elem.text:
                        transporter_genes.append(gene_name_elem.text)
            all_genes = list(set(target_genes + enzyme_genes + transporter_genes))
            if all_genes and atc_codes:
                drugs_data.append({
                    'drug_id': drug_id,
                    'name': name,
                    'atc_codes': atc_codes,
                    'target_genes': target_genes,
                    'enzyme_genes': enzyme_genes,
                    'transporter_genes': transporter_genes,
                    'all_genes': all_genes,
                    'num_targets': len(target_genes),
                    'num_enzymes': len(enzyme_genes),
                    'num_transporters': len(transporter_genes),
                    'num_total_genes': len(all_genes)
                })
            elem.clear()
            root.clear()
    return pd.DataFrame(drugs_data)

def combine_drug_sources(drugbank_df: pd.DataFrame, drugseqr_data: Dict) -> pd.DataFrame:
    """
    Combine DrugBank and DRUGseqr data.
    """
    combined_drugs = []
    for idx, row in drugbank_df.iterrows():
        combined_drugs.append({
            'drug_id': row['drug_id'],
            'drug_name': row['name'],
            'source': 'DrugBank',
            'target_genes': row['target_genes'],
            'enzyme_genes': row['enzyme_genes'],
            'transporter_genes': row['transporter_genes'],
            'all_genes': row['all_genes'],
            'num_genes': row['num_total_genes'],
            'atc_codes': row['atc_codes']
        })
    for sig_name, sig_data in drugseqr_data.items():
        if sig_data['num_genes'] > 0:
            combined_drugs.append({
                'drug_id': sig_data['cid'],
                'drug_name': sig_data['drug_name'],
                'source': 'DRUGseqr',
                'target_genes': sig_data['genes'],
                'enzyme_genes': [],
                'transporter_genes': [],
                'all_genes': sig_data['genes'],
                'num_genes': sig_data['num_genes'],
                'atc_codes': [],
                'expression_direction': sig_data['direction']
            })
    return pd.DataFrame(combined_drugs)

def execute(drugbank_xml: str, drugseqr_gmt: str, output_dir: Path) -> pd.DataFrame:
    """
    Execute Task 2: Extract and combine drug target genes, then generate required plots.
    """
    print("\n[Task 2] Extracting drug target genes from multiple sources...")
    drugseqr_data = parse_drugseqr_gmt(drugseqr_gmt)
    drugbank_df = parse_drugbank_xml(drugbank_xml)
    combined_df = combine_drug_sources(drugbank_df, drugseqr_data)
    # Save intermediate results
    drugbank_df.to_csv(output_dir / 'task2_drugbank_only.csv', index=False)
    drugseqr_df = pd.DataFrame([
        {
            'signature_name': k,
            'drug_name': v['drug_name'],
            'cid': v['cid'],
            'direction': v['direction'],
            'num_genes': v['num_genes']
        }
        for k, v in drugseqr_data.items()
    ])
    drugseqr_df.to_csv(output_dir / 'task2_drugseqr_only.csv', index=False)
    # Generate only the two required plots
    generate_task2_plots(combined_df, output_dir)
    print(f"\n  Saved intermediate files and plots.")
    return combined_df

if __name__ == "__main__":
    drugbank_xml = r"c:\Users\user\OneDrive\Desktop\My Files\LUH\Semester 5\DrugCategoryPrediction\drugbank_all_full_database.xml\full_database.xml"
    drugseqr_gmt = r"c:\Users\user\OneDrive\Desktop\My Files\LUH\Semester 5\DrugCategoryPrediction\dataset\DRUGseqr.gmt"
    output_dir = Path("../results")
    output_dir.mkdir(exist_ok=True)
    result = execute(drugbank_xml, drugseqr_gmt, output_dir)
    print(f"\n✓ Task 2 test completed. Total drugs: {len(result)}")
