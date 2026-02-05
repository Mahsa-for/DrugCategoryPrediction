"""
Real-World Example: Predict Categories for Novel Drug Candidates
This script demonstrates how to use the trained model for practical drug discovery scenarios
"""

import sys
import os

# Import necessary classes
from drug_category_predictor import CNSClassifier, DrugCategoryClassifier
from predict_drug_category import DrugCategoryPredictor
import pandas as pd

def predict_novel_compounds():
    """
    Scenario: A pharmaceutical company has developed novel compounds
    and wants to predict their therapeutic categories based on target genes
    """
    
    # Initialize predictor
    print("Loading trained models...")
    predictor = DrugCategoryPredictor(
        cns_model_path='results/cns_classifier.pkl',
        category_model_path='results/category_classifier.pkl',
        gene_expression_path='results/gene_expression.csv'
    )
    
    print("\n" + "=" * 80)
    print("REAL-WORLD SCENARIO: Novel Drug Candidate Prediction")
    print("=" * 80)
    
    # Define novel drug candidates with their target genes
    novel_compounds = {
        'Compound-001 (Alzheimer\'s)': ['BACE1', 'APP', 'MAPT', 'APOE'],
        'Compound-002 (Parkinson\'s)': ['SNCA', 'LRRK2', 'PARK7', 'PINK1'],
        'Compound-003 (Depression)': ['SLC6A4', 'HTR1A', 'HTR2A', 'MAOA'],
        'Compound-004 (Schizophrenia)': ['DRD2', 'DRD3', 'HTR2A', 'GRIN1'],
        'Compound-005 (Hypertension)': ['ACE', 'AGT', 'AGTR1', 'REN'],
        'Compound-006 (Diabetes)': ['INS', 'GCG', 'INSR', 'SLC2A2'],
        'Compound-007 (Cancer)': ['EGFR', 'TP53', 'BRCA1', 'ERBB2'],
        'Compound-008 (Pain)': ['OPRM1', 'OPRD1', 'OPRK1', 'TRPV1'],
        'Compound-009 (Asthma)': ['ADRB2', 'IL4', 'IL5', 'TSLP'],
        'Compound-010 (Inflammation)': ['TNF', 'IL6', 'IL1B', 'PTGS2']
    }
    
    print("\nPredicting therapeutic categories for 10 novel compounds...\n")
    
    results = []
    for compound_name, target_genes in novel_compounds.items():
        prediction = predictor.predict(target_genes, top_k=3)
        
        if 'error' not in prediction:
            results.append({
                'Compound': compound_name,
                'Target Genes': ', '.join(target_genes),
                'Genes Found': f"{prediction['num_genes_used']}/{len(target_genes)}",
                'CNS Active': '✓' if prediction['cns_classification']['is_cns_active'] else '✗',
                'CNS Prob': f"{prediction['cns_classification']['cns_probability']:.2%}",
                'Primary Category': prediction['primary_prediction']['category_name'],
                'Confidence': f"{prediction['primary_prediction']['probability']:.2%}",
                'Top-3 Categories': ' | '.join([
                    f"{p['category_code']}: {p['probability']:.1%}"
                    for p in prediction['top_predictions']
                ])
            })
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Display results
    print("\n" + "=" * 80)
    print("PREDICTION RESULTS")
    print("=" * 80)
    print()
    
    for idx, row in results_df.iterrows():
        print(f"📊 {row['Compound']}")
        print(f"   Target Genes: {row['Target Genes']}")
        print(f"   Genes Found: {row['Genes Found']}")
        print(f"   CNS Active: {row['CNS Active']} (Probability: {row['CNS Prob']})")
        print(f"   Primary Category: {row['Primary Category']}")
        print(f"   Confidence: {row['Confidence']}")
        print(f"   Top-3 Predictions: {row['Top-3 Categories']}")
        print()
    
    # Save results
    results_df.to_csv('results/novel_compound_predictions.csv', index=False)
    print("=" * 80)
    print(f"Results saved to: results/novel_compound_predictions.csv")
    print("=" * 80)
    
    # Summary statistics
    print("\n📈 SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total compounds analyzed: {len(results_df)}")
    print(f"CNS-active compounds: {results_df['CNS Active'].value_counts().get('✓', 0)}")
    print(f"Non-CNS compounds: {results_df['CNS Active'].value_counts().get('✗', 0)}")
    print()
    
    print("Category Distribution:")
    category_counts = results_df['Primary Category'].value_counts()
    for category, count in category_counts.items():
        print(f"  • {category}: {count}")
    
    print()
    print("Average Confidence:")
    avg_confidence = results_df['Confidence'].str.rstrip('%').astype(float).mean()
    print(f"  {avg_confidence:.2f}%")


def compare_with_known_drugs():
    """
    Compare predictions with known drugs from DrugBank
    """
    
    print("\n\n" + "=" * 80)
    print("VALIDATION: Comparing with Known Drugs")
    print("=" * 80)
    
    # Initialize predictor
    predictor = DrugCategoryPredictor(
        cns_model_path='results/cns_classifier.pkl',
        category_model_path='results/category_classifier.pkl',
        gene_expression_path='results/gene_expression.csv'
    )
    
    # Known drugs with their categories for validation
    known_drugs = {
        'Risperidone (Antipsychotic - N)': ['DRD2', 'HTR2A', 'ADRA1A'],
        'Atenolol (Beta-blocker - C)': ['ADRB1', 'ADRB2'],
        'Metformin (Antidiabetic - A)': ['PRKAA1', 'PRKAA2'],
        'Ibuprofen (NSAID - M)': ['PTGS1', 'PTGS2'],
        'Cetirizine (Antihistamine - R)': ['HRH1'],
    }
    
    print("\nValidating predictions against known drugs...\n")
    
    for drug_name, genes in known_drugs.items():
        result = predictor.predict(genes, top_k=3)
        
        actual_category = drug_name.split(' - ')[-1].rstrip(')')
        predicted_category = result['primary_prediction']['category_code']
        confidence = result['primary_prediction']['probability']
        
        match = "✓ CORRECT" if actual_category == predicted_category else "✗ INCORRECT"
        
        print(f"Drug: {drug_name}")
        print(f"  Targets: {', '.join(genes)}")
        print(f"  Predicted: {predicted_category} ({result['primary_prediction']['category_name']})")
        print(f"  Confidence: {confidence:.2%}")
        print(f"  Result: {match}")
        print()


def analyze_target_importance():
    """
    Analyze which target genes are most indicative of specific categories
    """
    
    print("\n" + "=" * 80)
    print("ANALYSIS: Target Gene Importance for Drug Categories")
    print("=" * 80)
    
    # Load drugs data
    drugs_df = pd.read_csv('results/drugs_data.csv')
    
    # Map genes to categories
    gene_category_map = {}
    
    for _, row in drugs_df.iterrows():
        if pd.notna(row['all_genes']) and pd.notna(row['atc_codes']):
            genes = eval(row['all_genes'])
            atc_codes = eval(row['atc_codes'])
            
            if atc_codes:
                primary_category = atc_codes[0][0]
                
                for gene in genes:
                    if gene not in gene_category_map:
                        gene_category_map[gene] = {}
                    
                    if primary_category not in gene_category_map[gene]:
                        gene_category_map[gene][primary_category] = 0
                    
                    gene_category_map[gene][primary_category] += 1
    
    # Find genes most specific to each category
    print("\nTop genes associated with each therapeutic category:\n")
    
    category_names = {
        'A': 'Alimentary/Metabolism',
        'B': 'Blood',
        'C': 'Cardiovascular',
        'D': 'Dermatologicals',
        'G': 'Genito-Urinary',
        'J': 'Anti-infectives',
        'L': 'Antineoplastic',
        'M': 'Musculo-Skeletal',
        'N': 'Nervous System',
        'R': 'Respiratory'
    }
    
    for category, name in category_names.items():
        # Find genes most associated with this category
        gene_scores = []
        
        for gene, categories in gene_category_map.items():
            if category in categories:
                total = sum(categories.values())
                specificity = categories[category] / total
                frequency = categories[category]
                
                if frequency >= 3:  # At least 3 occurrences
                    gene_scores.append((gene, specificity, frequency))
        
        # Sort by specificity
        gene_scores.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Category {category} - {name}:")
        for gene, specificity, frequency in gene_scores[:5]:
            print(f"  • {gene}: {specificity:.1%} specificity ({frequency} drugs)")
        print()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("DRUG CATEGORY PREDICTION - PRACTICAL APPLICATIONS")
    print("=" * 80)
    
    # Run all demonstrations
    predict_novel_compounds()
    compare_with_known_drugs()
    analyze_target_importance()
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE!")
    print("=" * 80)
    print("\nAll results have been saved to the results/ directory.")
    print("Check 'results/novel_compound_predictions.csv' for detailed predictions.")
