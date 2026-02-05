"""
Prediction Interface for Drug Category Prediction
Use trained models to predict drug categories based on target genes
"""

import joblib
import numpy as np
import pandas as pd
import sys
import os

# Import the classifier classes from the main module
from drug_category_predictor import (
    BrainGeneExpressionLoader, 
    DrugFeatureExtractor, 
    CNSClassifier, 
    DrugCategoryClassifier
)


class DrugCategoryPredictor:
    """Interface for predicting drug categories from target genes"""
    
    def __init__(self, cns_model_path, category_model_path, gene_expression_path):
        """
        Initialize predictor with trained models
        
        Args:
            cns_model_path: Path to trained CNS classifier
            category_model_path: Path to trained category classifier
            gene_expression_path: Path to gene expression CSV
        """
        print("Loading models...")
        self.cns_classifier = joblib.load(cns_model_path)
        self.category_classifier = joblib.load(category_model_path)
        
        print("Loading gene expression data...")
        self.gene_expression = pd.read_csv(gene_expression_path, index_col=0)
        self.feature_extractor = DrugFeatureExtractor(self.gene_expression)
        
        # ATC code descriptions
        self.atc_descriptions = {
            'A': 'Alimentary Tract and Metabolism',
            'B': 'Blood and Blood Forming Organs',
            'C': 'Cardiovascular System',
            'D': 'Dermatologicals',
            'G': 'Genito-Urinary System and Sex Hormones',
            'H': 'Systemic Hormonal Preparations',
            'J': 'Anti-infectives for Systemic Use',
            'L': 'Antineoplastic and Immunomodulating Agents',
            'M': 'Musculo-Skeletal System',
            'N': 'Nervous System',
            'P': 'Antiparasitic Products',
            'R': 'Respiratory System',
            'S': 'Sensory Organs',
            'V': 'Various'
        }
        
        print("Predictor ready!")
    
    def predict(self, gene_list, top_k=3):
        """
        Predict drug category for a list of target genes
        
        Args:
            gene_list: List of gene names (e.g., ['EGFR', 'TP53', 'BRCA1'])
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with predictions and probabilities
        """
        # Filter genes present in expression data
        available_genes = [g for g in gene_list if g in self.gene_expression.index]
        
        if not available_genes:
            return {
                'error': f'None of the provided genes found in brain expression data',
                'provided_genes': gene_list,
                'available_genes': list(self.gene_expression.index[:20])  # Show sample
            }
        
        # Extract features
        gene_expr = self.gene_expression.loc[available_genes]
        
        features = {
            'mean_expression': gene_expr.mean().values,
            'max_expression': gene_expr.max().values,
            'std_expression': gene_expr.std().values,
            'sum_expression': gene_expr.sum().values,
        }
        
        feature_vector = np.concatenate([
            features['mean_expression'],
            features['max_expression'],
            features['std_expression'],
            features['sum_expression']
        ]).reshape(1, -1)
        
        # Scale features
        feature_vector_cns = self.cns_classifier.scaler.transform(feature_vector)
        feature_vector_cat = self.category_classifier.scaler.transform(feature_vector)
        
        # Predict CNS vs Non-CNS
        cns_pred = self.cns_classifier.best_model.predict(feature_vector_cns)[0]
        cns_proba = self.cns_classifier.best_model.predict_proba(feature_vector_cns)[0]
        
        # Predict drug category
        category_pred = self.category_classifier.best_model.predict(feature_vector_cat)[0]
        category_proba = self.category_classifier.best_model.predict_proba(feature_vector_cat)[0]
        
        # Get top-k predictions
        top_indices = np.argsort(category_proba)[-top_k:][::-1]
        top_categories = [self.category_classifier.class_names[i] for i in top_indices]
        top_probas = [category_proba[i] for i in top_indices]
        
        # Format predictions
        predictions = []
        for cat, prob in zip(top_categories, top_probas):
            predictions.append({
                'category_code': cat,
                'category_name': self.atc_descriptions.get(cat, 'Unknown'),
                'probability': float(prob)
            })
        
        return {
            'input_genes': gene_list,
            'used_genes': available_genes,
            'num_genes_used': len(available_genes),
            'cns_classification': {
                'is_cns_active': bool(cns_pred == 1),
                'cns_probability': float(cns_proba[1]),
                'non_cns_probability': float(cns_proba[0])
            },
            'top_predictions': predictions,
            'primary_prediction': predictions[0] if predictions else None
        }
    
    def predict_batch(self, drugs_dict):
        """
        Predict categories for multiple drugs
        
        Args:
            drugs_dict: Dictionary of {drug_name: [gene_list]}
            
        Returns:
            Dictionary of predictions for each drug
        """
        results = {}
        for drug_name, gene_list in drugs_dict.items():
            print(f"Predicting for {drug_name}...")
            results[drug_name] = self.predict(gene_list)
        return results


def example_usage():
    """Example usage of the predictor"""
    
    # Load predictor
    predictor = DrugCategoryPredictor(
        cns_model_path='results/cns_classifier.pkl',
        category_model_path='results/category_classifier.pkl',
        gene_expression_path='results/gene_expression.csv'
    )
    
    # Example 1: Single drug prediction
    print("\n" + "=" * 60)
    print("Example 1: Predicting for a CNS drug (targets dopamine receptors)")
    print("=" * 60)
    
    genes_example1 = ['DRD2', 'DRD3', 'HTR2A', 'HTR2C']  # Typical antipsychotic targets
    result1 = predictor.predict(genes_example1, top_k=5)
    
    print(f"\nInput genes: {result1['input_genes']}")
    print(f"Genes found in database: {result1['used_genes']}")
    print(f"\nCNS Classification:")
    print(f"  Is CNS-active: {result1['cns_classification']['is_cns_active']}")
    print(f"  CNS probability: {result1['cns_classification']['cns_probability']:.4f}")
    print(f"\nTop predictions:")
    for i, pred in enumerate(result1['top_predictions'], 1):
        print(f"  {i}. {pred['category_name']} ({pred['category_code']}): {pred['probability']:.4f}")
    
    # Example 2: Cardiovascular drug
    print("\n" + "=" * 60)
    print("Example 2: Predicting for a cardiovascular drug")
    print("=" * 60)
    
    genes_example2 = ['ADRB1', 'ADRB2', 'ACE']  # Beta-blockers / ACE inhibitors
    result2 = predictor.predict(genes_example2, top_k=5)
    
    print(f"\nInput genes: {result2['input_genes']}")
    print(f"Genes found in database: {result2['used_genes']}")
    print(f"\nCNS Classification:")
    print(f"  Is CNS-active: {result2['cns_classification']['is_cns_active']}")
    print(f"  Non-CNS probability: {result2['cns_classification']['non_cns_probability']:.4f}")
    print(f"\nTop predictions:")
    for i, pred in enumerate(result2['top_predictions'], 1):
        print(f"  {i}. {pred['category_name']} ({pred['category_code']}): {pred['probability']:.4f}")
    
    # Example 3: Batch prediction
    print("\n" + "=" * 60)
    print("Example 3: Batch prediction for multiple drugs")
    print("=" * 60)
    
    drugs_batch = {
        'Antipsychotic': ['DRD2', 'DRD3', 'HTR2A'],
        'Beta-blocker': ['ADRB1', 'ADRB2'],
        'EGFR Inhibitor': ['EGFR', 'ERBB2'],
        'Opioid': ['OPRM1', 'OPRD1', 'OPRK1']
    }
    
    batch_results = predictor.predict_batch(drugs_batch)
    
    for drug_name, result in batch_results.items():
        print(f"\n{drug_name}:")
        if 'error' not in result:
            print(f"  Primary prediction: {result['primary_prediction']['category_name']}")
            print(f"  Probability: {result['primary_prediction']['probability']:.4f}")
            print(f"  CNS-active: {result['cns_classification']['is_cns_active']}")


if __name__ == "__main__":
    example_usage()
