"""
Task 2: Extract Drug Target Genes
Input: DrugBank + DRUGseqr.gmt datasets
Output: Targets/Enzymes/Transporters for each drug
Process: Transform (extract and combine)
"""

import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from typing import Dict, Set


def parse_drugseqr_gmt(gmt_path: str) -> Dict[str, Dict]:
    """
    Parse DRUGseqr.gmt file
    Format: Drug_name\tDrug_info\tGene1\tGene2\t...
    
    Returns:
        Dict with drug info and gene lists
    """
    print("\n  Parsing DRUGseqr.gmt...")
    
    drugs_data = {}
    
    with open(gmt_path, 'r') as f:
        for line_num, line in enumerate(tqdm(f, desc="    Reading DRUGseqr"), 1):
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            
            drug_name = parts[0]
            drug_info = parts[1]  # CID, InChI, SMILES
            genes = parts[2:]
            
            # Parse drug info
            info_parts = drug_info.split(';')
            cid = info_parts[0].replace('CID:', '') if len(info_parts) > 0 else ''
            
            # Determine if up or down regulation
            direction = 'unknown'
            if ' up' in drug_name.lower():
                direction = 'up'
            elif ' down' in drug_name.lower():
                direction = 'down'
            
            # Clean drug name
            clean_name = drug_name.split('_')[0] if '_' in drug_name else drug_name
            
            drugs_data[drug_name] = {
                'drug_name': clean_name,
                'cid': cid,
                'direction': direction,
                'genes': genes,
                'num_genes': len(genes)
            }
    
    print(f"    Parsed {len(drugs_data)} drug signatures")
    
    # Statistics
    up_count = sum(1 for d in drugs_data.values() if d['direction'] == 'up')
    down_count = sum(1 for d in drugs_data.values() if d['direction'] == 'down')
    print(f"    Up-regulated: {up_count}, Down-regulated: {down_count}")
    
    return drugs_data


def parse_drugbank_xml(xml_path: str) -> pd.DataFrame:
    """Parse DrugBank XML to extract drug-gene interactions"""
    print("\n  Parsing DrugBank XML...")
    
    namespace = {'db': 'http://www.drugbank.ca'}
    drugs_data = []
    
    context = ET.iterparse(xml_path, events=('start', 'end'))
    context = iter(context)
    event, root = next(context)
    
    drug_count = 0
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
            
            # Extract ATC codes
            atc_codes = []
            for atc in elem.findall('db:atc-codes/db:atc-code', namespace):
                code_attr = atc.get('code')
                if code_attr:
                    atc_codes.append(code_attr)
            
            # Extract target genes
            target_genes = []
            for target in elem.findall('db:targets/db:target', namespace):
                for poly in target.findall('db:polypeptide', namespace):
                    gene_name_elem = poly.find('db:gene-name', namespace)
                    if gene_name_elem is not None and gene_name_elem.text:
                        target_genes.append(gene_name_elem.text)
            
            # Extract enzyme genes
            enzyme_genes = []
            for enzyme in elem.findall('db:enzymes/db:enzyme', namespace):
                for poly in enzyme.findall('db:polypeptide', namespace):
                    gene_name_elem = poly.find('db:gene-name', namespace)
                    if gene_name_elem is not None and gene_name_elem.text:
                        enzyme_genes.append(gene_name_elem.text)
            
            # Extract transporter genes
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
                drug_count += 1
            
            elem.clear()
            root.clear()
    
    print(f"    Extracted {drug_count} drugs with genes and ATC codes")
    
    return pd.DataFrame(drugs_data)


def combine_drug_sources(drugbank_df: pd.DataFrame, drugseqr_data: Dict) -> pd.DataFrame:
    """Combine DrugBank and DRUGseqr data"""
    print("\n  Combining drug sources...")
    
    # Create a comprehensive drug-gene mapping
    combined_drugs = []
    
    # Process DrugBank drugs
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
    
    # Process DRUGseqr drugs
    drugseqr_with_genes = 0
    for sig_name, sig_data in drugseqr_data.items():
        if sig_data['num_genes'] > 0:
            combined_drugs.append({
                'drug_id': sig_data['cid'],
                'drug_name': sig_data['drug_name'],
                'source': 'DRUGseqr',
                'target_genes': sig_data['genes'],  # These are affected genes
                'enzyme_genes': [],
                'transporter_genes': [],
                'all_genes': sig_data['genes'],
                'num_genes': sig_data['num_genes'],
                'atc_codes': [],  # DRUGseqr doesn't have ATC codes
                'expression_direction': sig_data['direction']
            })
            drugseqr_with_genes += 1
    
    combined_df = pd.DataFrame(combined_drugs)
    
    print(f"    DrugBank drugs: {len(drugbank_df)}")
    print(f"    DRUGseqr signatures: {drugseqr_with_genes}")
    print(f"    Total combined entries: {len(combined_df)}")
    
    # Statistics
    print(f"\n  Gene statistics:")
    print(f"    Mean genes per drug: {combined_df['num_genes'].mean():.1f}")
    print(f"    Median genes per drug: {combined_df['num_genes'].median():.1f}")
    print(f"    Max genes per drug: {combined_df['num_genes'].max()}")
    
    # Get all unique genes
    all_genes_set = set()
    for genes in combined_df['all_genes']:
        if isinstance(genes, list):
            all_genes_set.update(genes)
    print(f"    Total unique genes: {len(all_genes_set)}")
    
    return combined_df


def execute(drugbank_xml: str, drugseqr_gmt: str, output_dir: Path) -> pd.DataFrame:
    """
    Execute Task 2: Extract and combine drug target genes
    
    Args:
        drugbank_xml: Path to DrugBank XML file
        drugseqr_gmt: Path to DRUGseqr.gmt file
        output_dir: Directory to save outputs
    
    Returns:
        DataFrame with drug-gene relationships from both sources
    """
    print("\n[Task 2] Extracting drug target genes from multiple sources...")
    
    # Parse DRUGseqr
    drugseqr_data = parse_drugseqr_gmt(drugseqr_gmt)
    
    # Parse DrugBank
    drugbank_df = parse_drugbank_xml(drugbank_xml)
    
    # Combine sources
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
    
    print(f"\n  Saved intermediate files:")
    print(f"    - task2_drugbank_only.csv")
    print(f"    - task2_drugseqr_only.csv")
    
    return combined_df


if __name__ == "__main__":
    # Test the task
    drugbank_xml = r"c:\Users\user\OneDrive\Desktop\My Files\LUH\Semester 5\DrugCategoryPrediction\drugbank_all_full_database.xml\full_database.xml"
    drugseqr_gmt = r"c:\Users\user\OneDrive\Desktop\My Files\LUH\Semester 5\DrugCategoryPrediction\dataset\DRUGseqr.gmt"
    output_dir = Path("../results")
    output_dir.mkdir(exist_ok=True)
    
    result = execute(drugbank_xml, drugseqr_gmt, output_dir)
    print(f"\n✓ Task 2 test completed. Total drugs: {len(result)}")
