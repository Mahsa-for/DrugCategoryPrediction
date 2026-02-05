"""
Interactive Drug Category Prediction Interface
User-friendly interface for predicting drug categories from target genes
"""

import joblib
import numpy as np
import pandas as pd
import os
from pathlib import Path


class InteractiveDrugPredictor:
    """Interactive interface for drug category prediction"""
    
    def __init__(self, models_dir='results'):
        """Initialize with trained models from NeSy system"""
        self.models_dir = Path(models_dir)
        self.predictor = None
        self.gene_expression = None
        self.classifier_model = None
        self.label_encoder = None
        self.feature_scaler = None
        
        # ATC descriptions
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
        
        self.load_models()
    
    def load_models(self):
        """Load trained models and data from NeSy results"""
        print("\n" + "="*70)
        print("LOADING MODELS")
        print("="*70)
        
        try:
            # Load brain expression data
            expr_file = self.models_dir / 'task1_brain_expression.csv'
            if expr_file.exists():
                self.gene_expression = pd.read_csv(expr_file, index_col=0)
                print(f"[OK] Gene expression data loaded: {self.gene_expression.shape[0]} genes x {self.gene_expression.shape[1]} regions")
            else:
                print(f"[ERROR] Gene expression file not found: {expr_file}")
                return False
            
            # Load trained classifier
            model_file = self.models_dir / 'task5_classifier_model.pkl'
            if model_file.exists():
                self.classifier_model = joblib.load(model_file)
                print(f"[OK] Classifier model loaded")
            else:
                print(f"[ERROR] Classifier model not found: {model_file}")
                return False
            
            # Load label encoder
            encoder_file = self.models_dir / 'task5_label_encoder.pkl'
            if encoder_file.exists():
                self.label_encoder = joblib.load(encoder_file)
                print(f"[OK] Label encoder loaded: {len(self.label_encoder.classes_)} categories")
            else:
                print(f"[ERROR] Label encoder not found: {encoder_file}")
                return False
            
            # Load feature scaler
            scaler_file = self.models_dir / 'task4_feature_scaler.pkl'
            if scaler_file.exists():
                self.feature_scaler = joblib.load(scaler_file)
                print(f"[OK] Feature scaler loaded")
            else:
                print(f"[ERROR] Feature scaler not found: {scaler_file}")
                return False
            
            print("\n[OK] All models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"\n[ERROR] Error loading models: {e}")
            return False
    
    def extract_features(self, gene_list):
        """Extract features from gene list"""
        # Filter genes present in expression data
        available_genes = [g.upper() for g in gene_list if g.upper() in self.gene_expression.index]
        
        if not available_genes:
            return None, gene_list, []
        
        # Get expression profiles
        gene_expr = self.gene_expression.loc[available_genes]
        
        # Create features matching Task 4 format
        n_regions = self.gene_expression.shape[1]
        region_names = self.gene_expression.columns.tolist()
        
        features = []
        
        # 1. Mean expression per brain region
        features.extend(gene_expr.mean(axis=0).values)
        
        # 2. Max expression per brain region
        features.extend(gene_expr.max(axis=0).values)
        
        # 3. Standard deviation per brain region
        features.extend(gene_expr.std(axis=0).values)
        
        # 4. Overall statistics
        features.append(gene_expr.values.mean())      # Overall mean
        features.append(gene_expr.values.std())       # Overall std
        features.append(gene_expr.values.max())       # Overall max
        features.append(len(available_genes))         # Number of genes
        features.append(len(available_genes) / len(gene_list) if gene_list else 0)  # Coverage ratio
        
        feature_vector = np.array(features).reshape(1, -1)
        
        return feature_vector, gene_list, available_genes
    
    def predict(self, gene_list, top_k=5):
        """
        Predict drug category from target genes
        
        Args:
            gene_list: List of gene names
            top_k: Number of top predictions
            
        Returns:
            Dictionary with predictions
        """
        if not self.classifier_model or not self.label_encoder:
            return {'error': 'Models not loaded. Run NeSy system first.'}
        
        # Extract features
        feature_vector, input_genes, available_genes = self.extract_features(gene_list)
        
        if feature_vector is None:
            return {
                'error': f'None of the {len(gene_list)} provided genes found in brain expression database',
                'input_genes': input_genes,
                'database_genes_sample': list(self.gene_expression.index[:10])
            }
        
        # Scale features
        feature_vector_scaled = self.feature_scaler.transform(feature_vector)
        
        # Get predictions
        y_pred = self.classifier_model.predict(feature_vector_scaled)[0]
        
        # Get probabilities if available
        if hasattr(self.classifier_model, 'predict_proba'):
            probas = self.classifier_model.predict_proba(feature_vector_scaled)[0]
        else:
            # Fallback for models without predict_proba
            probas = np.zeros(len(self.label_encoder.classes_))
            probas[y_pred] = 1.0
        
        # Get top-k predictions
        top_k = min(top_k, len(self.label_encoder.classes_))
        top_indices = np.argsort(probas)[-top_k:][::-1]
        
        predictions = []
        for idx in top_indices:
            category_code = self.label_encoder.classes_[idx]
            probability = probas[idx]
            predictions.append({
                'category_code': category_code,
                'category_name': self.atc_descriptions.get(category_code, 'Unknown Category'),
                'probability': float(probability)
            })
        
        # Compute brain evidence metrics
        brain_evidence = self._compute_brain_evidence(available_genes)
        
        return {
            'success': True,
            'input_genes': input_genes,
            'genes_found': len(available_genes),
            'genes_used': available_genes,
            'primary_prediction': predictions[0] if predictions else None,
            'top_predictions': predictions,
            'brain_evidence': brain_evidence,
            'brain_evidence': brain_evidence
        }
    
    def _compute_brain_evidence(self, available_genes):
        """Compute brain evidence metrics for given genes"""
        try:
            from src.metrics.brain_evidence import BrainEvidenceMetrics
            
            # Get expression data for available genes
            if not available_genes:
                return None
            
            gene_expr_data = self.gene_expression.loc[available_genes]
            
            # Define brain regions and body tissues
            # Assuming columns contain region names
            all_regions = self.gene_expression.columns.tolist()
            
            # Heuristic: identify brain regions (contains 'brain', 'cortex', 'hippocampus', etc.)
            brain_keywords = ['brain', 'cortex', 'hippocampus', 'striatum', 'cerebellum', 
                             'amygdala', 'thalamus', 'hypothalamus', 'midbrain', 'pons', 'medulla']
            brain_regions = [r for r in all_regions if any(kw in r.lower() for kw in brain_keywords)]
            
            # If no brain regions identified by keywords, use all regions as brain (default behavior)
            if not brain_regions:
                brain_regions = all_regions
            
            body_tissues = [r for r in all_regions if r not in brain_regions]
            
            # Build gene expression dictionary in the format expected by BrainEvidenceMetrics
            gene_expression_dict = {}
            for gene in available_genes:
                gene_expression_dict[gene] = {}
                for region in all_regions:
                    gene_expression_dict[gene][region] = float(gene_expr_data.loc[gene, region])
            
            # Compute metrics
            metrics = BrainEvidenceMetrics(tau_strength=0.3, tau_ratio=0.6)
            summary = metrics.evidence_summary(gene_expression_dict, brain_regions, body_tissues)
            
            return summary
            
        except Exception as e:
            print(f"Warning: Could not compute brain evidence metrics: {e}")
            return None
    
    def display_prediction(self, result):
        """Display prediction result in user-friendly format"""
        if 'error' in result:
            print(f"\n[ERROR] {result['error']}")
            if 'database_genes_sample' in result:
                print(f"\nExample genes in database:")
                for gene in result['database_genes_sample']:
                    print(f"  - {gene}")
            return
        
        print("\n" + "="*70)
        print("PREDICTION RESULTS")
        print("="*70)
        
        # Input summary
        print(f"\n📥 INPUT:")
        print(f"  Genes provided: {result['genes_found']}/{len(result['input_genes'])}")
        print(f"  Genes used: {', '.join(result['genes_used'])}")
        
        # Primary prediction
        primary = result['primary_prediction']
        print(f"\n🎯 PRIMARY PREDICTION:")
        print(f"  Category: {primary['category_name']}")
        print(f"  Code: {primary['category_code']}")
        print(f"  Confidence: {primary['probability']:.2%}")
        
        # Top predictions
        print(f"\n📊 TOP {len(result['top_predictions'])} PREDICTIONS:")
        for i, pred in enumerate(result['top_predictions'], 1):
            confidence_bar = "█" * int(pred['probability'] * 20) + "░" * (20 - int(pred['probability'] * 20))
            print(f"  {i}. [{confidence_bar}] {pred['category_name']} ({pred['category_code']})")
            print(f"     Confidence: {pred['probability']:.2%}")


def main():
    """Main interactive interface"""
    
    # Initialize
    predictor = InteractiveDrugPredictor()
    
    # Check if models are loaded
    if not predictor.classifier_model:
        print("\n[ERROR] Models not loaded. Please run 'python nesy_system.py' first.")
        return
    
    print("\n" + "="*70)
    print("INTERACTIVE DRUG CATEGORY PREDICTION")
    print("="*70)
    print("\nPredicts therapeutic drug categories based on target genes")
    print("\nCommands:")
    print("  'predict' - Predict category for genes")
    print("  'batch'   - Predict for multiple drugs")
    print("  'examples'- Show example predictions")
    print("  'exit'    - Exit program")
    
    while True:
        print("\n" + "-"*70)
        command = input("Enter command (predict/batch/examples/exit): ").strip().lower()
        
        if command == 'exit':
            print("\n[OK] Goodbye!")
            break
        
        elif command == 'predict':
            print("\nEnter target gene names (comma-separated)")
            print("Example: TP53, EGFR, BRCA1")
            genes_input = input("Genes: ").strip()
            
            if not genes_input:
                print("[ERROR] No genes provided")
                continue
            
            # Parse genes
            genes = [g.strip().upper() for g in genes_input.split(',')]
            genes = [g for g in genes if g]  # Remove empty strings
            
            if not genes:
                print("[ERROR] No valid genes provided")
                continue
            
            # Get top_k
            try:
                top_k_input = input("Top-k predictions (default 5): ").strip()
                top_k = int(top_k_input) if top_k_input else 5
            except ValueError:
                top_k = 5
            
            # Predict
            result = predictor.predict(genes, top_k=top_k)
            predictor.display_prediction(result)
        
        elif command == 'batch':
            print("\nEnter drugs in format: drug_name:gene1,gene2,gene3")
            print("Example:")
            print("  Aspirin:PTGS1,PTGS2")
            print("  Metformin:AMP,GATA4")
            print("\nEnter 'done' when finished")
            
            drugs = {}
            while True:
                drug_input = input("Drug: ").strip()
                
                if drug_input.lower() == 'done':
                    break
                
                if ':' not in drug_input:
                    print("[ERROR] Invalid format. Use: drug_name:gene1,gene2")
                    continue
                
                drug_name, genes_str = drug_input.split(':', 1)
                genes = [g.strip().upper() for g in genes_str.split(',')]
                genes = [g for g in genes if g]
                
                if genes:
                    drugs[drug_name.strip()] = genes
            
            if drugs:
                print(f"\n[OK] Predicting for {len(drugs)} drugs...")
                for drug_name, genes in drugs.items():
                    print(f"\n{'='*70}")
                    print(f"Drug: {drug_name}")
                    result = predictor.predict(genes, top_k=3)
                    predictor.display_prediction(result)
        
        elif command == 'examples':
            print("\n" + "="*70)
            print("EXAMPLE PREDICTIONS")
            print("="*70)
            
            examples = {
                'Antipsychotic (Dopamine antagonist)': ['DRD2', 'DRD3', 'HTR2A'],
                'Beta-blocker (Cardiovascular)': ['ADRB1', 'ADRB2'],
                'Antibiotic (Bacterial DNA)': ['GYRA', 'GYRB', 'TOPOISOMERASE'],
                'Cancer drug (EGFR inhibitor)': ['EGFR', 'ERBB2'],
                'Opioid (Pain management)': ['OPRM1', 'OPRD1']
            }
            
            for description, genes in examples.items():
                print(f"\n{'─'*70}")
                print(f"Example: {description}")
                result = predictor.predict(genes, top_k=3)
                predictor.display_prediction(result)
        
        else:
            print("[ERROR] Unknown command. Try 'predict', 'batch', 'examples', or 'exit'")


if __name__ == "__main__":
    main()
