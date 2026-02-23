"""
Task 3: Fetch ATC Hierarchy
Input: DrugBank
Output: ATC therapeutic categories
Process: Transform (extract hierarchical categories)
"""

import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from typing import Dict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ATC Code Level Descriptions
ATC_LEVEL_NAMES = {
    1: 'Anatomical Main Group',
    2: 'Therapeutic Subgroup',
    3: 'Pharmacological Subgroup',
    4: 'Chemical Subgroup',
    5: 'Chemical Substance'
}

# First level ATC codes
ATC_MAIN_GROUPS = {
    'A': 'Alimentary tract and metabolism',
    'B': 'Blood and blood forming organs',
    'C': 'Cardiovascular system',
    'D': 'Dermatologicals',
    'G': 'Genito-urinary system and sex hormones',
    'H': 'Systemic hormonal preparations',
    'J': 'Antiinfectives for systemic use',
    'L': 'Antineoplastic and immunomodulating agents',
    'M': 'Musculo-skeletal system',
    'N': 'Nervous system',
    'P': 'Antiparasitic products',
    'R': 'Respiratory system',
    'S': 'Sensory organs',
    'V': 'Various'
}


def parse_atc_code(atc_code: str) -> Dict[str, str]:
    """
    Parse ATC code into hierarchical levels
    Example: N06AB05
    - N: Nervous system
    - N06: Psychoanaleptics
    - N06A: Antidepressants
    - N06AB: Selective serotonin reuptake inhibitors
    - N06AB05: Paroxetine
    """
    levels = {}
    
    if len(atc_code) >= 1:
        levels['level1'] = atc_code[0]
        levels['level1_name'] = ATC_MAIN_GROUPS.get(atc_code[0], 'Unknown')
    
    if len(atc_code) >= 3:
        levels['level2'] = atc_code[:3]
    
    if len(atc_code) >= 4:
        levels['level3'] = atc_code[:4]
    
    if len(atc_code) >= 5:
        levels['level4'] = atc_code[:5]
    
    if len(atc_code) == 7:
        levels['level5'] = atc_code
    
    levels['full_code'] = atc_code
    
    return levels


def execute(drugbank_xml: str, output_dir: Path) -> pd.DataFrame:
    """
    Execute Task 3: Extract ATC hierarchical categories
    
    Args:
        drugbank_xml: Path to DrugBank XML file
        output_dir: Directory to save outputs
    
    Returns:
        DataFrame with drug ATC hierarchical information
    """
    print("\n[Task 3] Extracting ATC hierarchical categories...")
    
    namespace = {'db': 'http://www.drugbank.ca'}
    atc_data = []
    
    context = ET.iterparse(drugbank_xml, events=('start', 'end'))
    context = iter(context)
    event, root = next(context)
    
    drug_count = 0
    drugs_with_atc = 0
    
    print("\n  Parsing DrugBank for ATC codes...")
    for event, elem in tqdm(context, desc="    Processing drugs"):
        if event == 'end' and elem.tag == '{http://www.drugbank.ca}drug':
            drug_id_elem = elem.find('db:drugbank-id[@primary="true"]', namespace)
            name_elem = elem.find('db:name', namespace)
            
            if drug_id_elem is None or name_elem is None:
                elem.clear()
                root.clear()
                continue
            
            drug_id = drug_id_elem.text
            name = name_elem.text
            drug_count += 1
            
            # Extract ATC codes and levels
            atc_codes = []
            for atc in elem.findall('db:atc-codes/db:atc-code', namespace):
                code_attr = atc.get('code')
                if code_attr:
                    atc_codes.append(code_attr)
                    
                    # Parse hierarchical levels
                    levels = parse_atc_code(code_attr)
                    
                    # Get level name if available
                    level_elem = atc.find('db:level', namespace)
                    level_name = level_elem.text if level_elem is not None else None
                    
                    atc_data.append({
                        'drug_id': drug_id,
                        'drug_name': name,
                        'atc_code': code_attr,
                        'atc_level1': levels.get('level1', ''),
                        'atc_level1_name': levels.get('level1_name', ''),
                        'atc_level2': levels.get('level2', ''),
                        'atc_level3': levels.get('level3', ''),
                        'atc_level4': levels.get('level4', ''),
                        'atc_level5': levels.get('level5', ''),
                        'level_name': level_name
                    })
            
            if atc_codes:
                drugs_with_atc += 1
            
            elem.clear()
            root.clear()
    
    print(f"\n  Statistics:")
    print(f"    Total drugs processed: {drug_count}")
    print(f"    Drugs with ATC codes: {drugs_with_atc} ({drugs_with_atc/drug_count*100:.1f}%)")
    
    # Create DataFrame
    atc_df = pd.DataFrame(atc_data)
    print(f"    Total ATC code assignments: {len(atc_df)}")
    
    # Add primary category (use level 1 as primary)
    atc_df['primary_category'] = atc_df['atc_level1']
    
    # Statistics by level
    print(f"\n  ATC Level 1 (Main Groups) distribution:")
    level1_counts = atc_df['atc_level1'].value_counts()
    for code, count in level1_counts.items():
        name = ATC_MAIN_GROUPS.get(code, 'Unknown')
        print(f"    {code} - {name}: {count} ({count/len(atc_df)*100:.1f}%)")

    # --- Plots for report ---
    # 1. ATC Main Group (Level 1) distribution
    plt.figure(figsize=(10, 6))
    level1_counts = atc_df['atc_level1'].value_counts().sort_index()
    labels = [f"{code} - {ATC_MAIN_GROUPS.get(code, 'Unknown')}" for code in level1_counts.index]
    plt.bar(labels, level1_counts.values, color='cornflowerblue')
    plt.title('ATC Main Group (Level 1) Distribution')
    plt.ylabel('Number of Assignments')
    plt.xlabel('ATC Main Group')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'task3_atc_level1_distribution.png')
    plt.close()

    # 2. Number of ATC codes per drug
    drugs_multi_atc = atc_df.groupby('drug_id').size()
    plt.figure(figsize=(8, 5))
    drugs_multi_atc.hist(bins=range(1, drugs_multi_atc.max()+2), color='salmon', alpha=0.8)
    plt.title('Number of ATC Codes per Drug')
    plt.xlabel('ATC Codes per Drug')
    plt.ylabel('Number of Drugs')
    plt.tight_layout()
    plt.savefig(output_dir / 'task3_atc_codes_per_drug.png')
    plt.close()

    # 3. Unique categories per ATC level
    level_stats = []
    for level in ['atc_level1', 'atc_level2', 'atc_level3']:
        unique_count = atc_df[level].nunique()
        level_stats.append({
            'level': level,
            'unique_categories': unique_count
        })
    level_stats_df = pd.DataFrame(level_stats)
    level_stats_df.to_csv(output_dir / 'task3_atc_level_statistics.csv', index=False)
    plt.figure(figsize=(6, 4))
    plt.bar(level_stats_df['level'], level_stats_df['unique_categories'], color='mediumseagreen')
    plt.title('Unique Categories per ATC Level')
    plt.xlabel('ATC Level')
    plt.ylabel('Number of Unique Categories')
    plt.tight_layout()
    plt.savefig(output_dir / 'task3_unique_categories_per_level.png')
    plt.close()

    print(f"\n  Saved level statistics to: task3_atc_level_statistics.csv")
    return atc_df


if __name__ == "__main__":
    # Test the task
    drugbank_xml = r"c:\Users\user\OneDrive\Desktop\My Files\LUH\Semester 5\DrugCategoryPrediction\drugbank_all_full_database.xml\full_database.xml"
    output_dir = Path("../results")
    output_dir.mkdir(exist_ok=True)
    
    result = execute(drugbank_xml, output_dir)
    print(f"\n✓ Task 3 test completed. ATC assignments: {len(result)}")
