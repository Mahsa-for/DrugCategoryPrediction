"""
NeSy System for Drug Category Prediction based on Gene Expression
Following the gene-correlation.json design pattern
"""

import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

# Import all task modules
from tasks import (
    task1_brain_gene_expression,
    task2_extract_drug_targets,
    task3_fetch_atc_hierarchy,
    task4_integrate_gene_signatures,
    task5a_cns_classifier,
    task5b_train_classifier,
    task6_predict_evaluate
)

# Import validation
from drug_category_predictor import DatasetValidator


class NeSyDrugPredictionSystem:
    """
    Neuro-Symbolic System for Drug Category Prediction
    
    Based on the following architecture:
    1. Gene Expression by Brain Cell Cluster Type (scRNA data)
    2. Extract Drug Target Genes (DrugBank + DRUGseqr)
    3. Fetch ATC Hierarchy (Therapeutic categories)
    4. Integrate Gene Signatures (Embedding)
    5. Train Classification Model (Infer drug categories)
    6. Predict with Top-k probabilities
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results_dir = Path(config['results_dir'])
        self.results_dir.mkdir(exist_ok=True)
        
        # Storage for intermediate results
        self.brain_expression = None
        self.drug_targets = None
        self.atc_hierarchy = None
        self.integrated_features = None
        self.cns_classifier = None
        self.cns_scaler = None
        self.classifier_model = None
        
        # Evaluation metrics storage
        self.metrics = {}
        self.cns_metrics = {}
        
    def validate_datasets(self) -> bool:
        """Step 0: Validate all required datasets"""
        print("\n" + "="*80)
        print("STEP 0: DATASET VALIDATION")
        print("="*80)
        
        validator = DatasetValidator(
            self.config['drugbank_xml'],
            self.config['cluster_type_tsv']
        )
        can_proceed, validation_results = validator.validate_all()
        
        # Check DRUGseqr.gmt
        if os.path.exists(self.config['drugseqr_gmt']):
            print(f"\n  ✓ DRUGseqr.gmt found: {self.config['drugseqr_gmt']}")
            file_size = os.path.getsize(self.config['drugseqr_gmt']) / (1024**2)
            print(f"    File size: {file_size:.2f} MB")
        else:
            print(f"\n  ✗ DRUGseqr.gmt NOT FOUND: {self.config['drugseqr_gmt']}")
            can_proceed = False
        
        return can_proceed
    
    def task1_brain_regions(self) -> pd.DataFrame:
        """
        Task 1: Gene Expression by Brain Cell Cluster Type
        Input: scRNA Brain Genes dataset
        Output: Expression of genes in each brain cell cluster type
        Process: Train (analyze expression patterns by cell cluster type)
        """
        print("\n" + "="*80)
        print("TASK 1: GENE EXPRESSION BY BRAIN CELL CLUSTER TYPE")
        print("="*80)
        
        self.brain_expression = task1_brain_gene_expression.execute(
            cluster_type_path=self.config['cluster_type_tsv'],
            output_dir=self.results_dir
        )
        
        # Save task result
        output_file = self.results_dir / 'task1_brain_expression.csv'
        self.brain_expression.to_csv(output_file)
        print(f"\n✓ Task 1 completed. Results saved to: {output_file}")
        
        return self.brain_expression
    
    def task2_drug_targets(self) -> pd.DataFrame:
        """
        Task 2: Extract Drug Target Genes
        Input: DrugBank + DRUGseqr datasets
        Output: Targets/Enzymes/Transporters for each drug
        Process: Transform (extract and combine)
        """
        print("\n" + "="*80)
        print("TASK 2: EXTRACT DRUG TARGET GENES")
        print("="*80)
        
        self.drug_targets = task2_extract_drug_targets.execute(
            drugbank_xml=self.config['drugbank_xml'],
            drugseqr_gmt=self.config['drugseqr_gmt'],
            output_dir=self.results_dir
        )
        
        # Save task result
        output_file = self.results_dir / 'task2_drug_targets.csv'
        self.drug_targets.to_csv(output_file, index=False)
        print(f"\n✓ Task 2 completed. Results saved to: {output_file}")
        
        return self.drug_targets
    
    def task3_atc_categories(self) -> pd.DataFrame:
        """
        Task 3: Fetch ATC Hierarchy
        Input: DrugBank
        Output: ATC therapeutic categories
        Process: Transform (extract hierarchical categories)
        """
        print("\n" + "="*80)
        print("TASK 3: FETCH ATC HIERARCHY")
        print("="*80)
        
        self.atc_hierarchy = task3_fetch_atc_hierarchy.execute(
            drugbank_xml=self.config['drugbank_xml'],
            output_dir=self.results_dir
        )
        
        # Save task result
        output_file = self.results_dir / 'task3_atc_hierarchy.csv'
        self.atc_hierarchy.to_csv(output_file, index=False)
        print(f"\n✓ Task 3 completed. Results saved to: {output_file}")
        
        return self.atc_hierarchy
    
    def task4_integration(self) -> Tuple[np.ndarray, List[int]]:
        """
        Task 4: Integrate Gene Signatures
        Input: Brain cell cluster type expression + Drug targets + Gene signatures
        Output: Integrated feature embeddings
        Process: Embedding (combine all information)
        """
        print("\n" + "="*80)
        print("TASK 4: INTEGRATE GENE SIGNATURES (by cell cluster type)")
        print("="*80)
        
        # Task 4 returns only features and valid_indices, but feature names are saved in a CSV in task4_integrate_gene_signatures
        self.integrated_features, valid_indices = task4_integrate_gene_signatures.execute(
            brain_expression=self.brain_expression,
            drug_targets=self.drug_targets,
            output_dir=self.results_dir
        )

        # Load feature names from the feature importance CSV (created in Task 4)
        import pandas as pd
        feature_importance_path = self.results_dir / 'task4_feature_importance.csv'
        if feature_importance_path.exists():
            feature_importance_df = pd.read_csv(feature_importance_path)
            self.feature_names = feature_importance_df['feature_name'].tolist()
        else:
            raise FileNotFoundError(f"Feature importance file not found: {feature_importance_path}")

        # Save task result
        np.save(self.results_dir / 'task4_integrated_features.npy', self.integrated_features)
        np.save(self.results_dir / 'task4_valid_indices.npy', valid_indices)
        print(f"\n✓ Task 4 completed. Feature shape: {self.integrated_features.shape}")

        return self.integrated_features, valid_indices
    
    def task5a_cns_classification(self, X: np.ndarray, valid_indices: List[int]) -> Tuple[Any, StandardScaler]:
        """
        Task 5a: CNS vs Non-CNS Classification (Stage 1)
        Input: Integrated features (including cell cluster type expression) + ATC hierarchy
        Output: Trained CNS classifier
        Process: Infer (learn CNS-relevant patterns)
        """
        print("\n" + "="*80)
        print("TASK 5a: CNS vs NON-CNS CLASSIFICATION (Stage 1)")
        print("="*80)
        
        # Ensure feature_names is available and includes 'BES' and 'BSR'
        if hasattr(self, 'feature_names'):
            feature_names = self.feature_names
        else:
            # Fallback: try to infer from X if it's a DataFrame, else raise error
            if hasattr(X, 'columns'):
                feature_names = list(X.columns)
            else:
                raise ValueError("feature_names must be provided as an attribute of the system or as columns of X.")

        if 'BES' not in feature_names or 'BSR' not in feature_names:
            raise ValueError("feature_names must include 'BES' and 'BSR'. Got: {}".format(feature_names))

        self.cns_classifier, self.cns_scaler, cns_metrics = task5a_cns_classifier.execute(
            X=X,
            drug_targets=self.drug_targets,
            atc_hierarchy=self.atc_hierarchy,
            valid_indices=valid_indices,
            output_dir=self.results_dir,
            feature_names=feature_names
        )
        
        # Store CNS metrics
        self.cns_metrics = cns_metrics
        
        print(f"\n✓ Task 5a completed. CNS Classifier: {cns_metrics['best_model']}")
        print(f"   Accuracy: {cns_metrics['train_accuracy']:.4f}")
        print(f"   CNS drugs: {cns_metrics['num_cns_samples']}, Non-CNS drugs: {cns_metrics['num_non_cns_samples']}")
        
        return self.cns_classifier, self.cns_scaler
    
    def task5b_train_classifier_model(self, X_train, y_train) -> Any:
        """
        Task 5: Train Drug Category Classification Model
        Input: Integrated features + ATC categories
        Output: Trained classification model
        Process: Infer (learn drug category patterns)
        """
        print("\n" + "="*80)
        print("TASK 5: TRAIN DRUG CATEGORY CLASSIFICATION MODEL")
        print("="*80)
        
        # Prepare data with labels and filter
        from sklearn.preprocessing import LabelEncoder
        
        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_train)
        
        self.label_encoder = le
        
        # Use task5 to train
        print("\n  Training with pre-prepared labels...")
        print(f"    Training samples: {len(X_train)}")
        print(f"    Number of classes: {len(le.classes_)}")
        print(f"    Classes: {', '.join(le.classes_)}")
        
        # Import training functions
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.metrics import accuracy_score, f1_score
        
        # Train models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        }
        
        results = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in models.items():
            print(f"\n    Training {name}...")
            cv_scores = cross_val_score(model, X_train, y_encoded, cv=cv, scoring='accuracy')
            model.fit(X_train, y_encoded)
            
            train_pred = model.predict(X_train)
            train_acc = accuracy_score(y_encoded, train_pred)
            
            results[name] = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'train_accuracy': train_acc
            }
            
            print(f"      CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Select best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['cv_mean'])
        self.classifier_model = results[best_model_name]['model']
        
        # Store training metrics
        self.metrics['training'] = {
            'best_model': best_model_name,
            'cv_accuracy': results[best_model_name]['cv_mean'],
            'train_accuracy': results[best_model_name]['train_accuracy'],
            'num_classes': len(le.classes_),
            'class_names': le.classes_.tolist()
        }
        
        # Patch model.classes_ to ATC codes before saving
        if hasattr(self.classifier_model, 'classes_'):
            self.classifier_model.classes_ = le.classes_
        joblib.dump(self.classifier_model, self.results_dir / 'task5_classifier_model.pkl')
        joblib.dump(self.label_encoder, self.results_dir / 'task5_label_encoder.pkl')
        print(f"\n✓ Task 5 completed. Best model: {best_model_name}")
        
        return self.classifier_model
    
    def task6_predict(self, model, X_test, y_test_raw, top_k=5) -> Dict[str, Any]:
        """
        Task 6: Predict Drug Categories
        Input: Trained model + Test data
        Output: Top-k predicted therapeutic classes + probabilities
        Process: Deduce (predict with confidence scores)
        """
        print("\n" + "="*80)
        print("TASK 6: PREDICTION AND EVALUATION")
        print("="*80)
        
        # Encode test labels
        y_test = self.label_encoder.transform(y_test_raw)
        
        predictions, eval_metrics = task6_predict_evaluate.execute(
            model=model,
            X_test=X_test,
            y_test=y_test,
            top_k=top_k,
            output_dir=self.results_dir
        )
        
        # Store evaluation metrics
        self.metrics['evaluation'] = eval_metrics
        
        # Save predictions
        predictions_df = pd.DataFrame(predictions)
        predictions_df.to_csv(self.results_dir / 'task6_predictions.csv', index=False)
        print(f"\n✓ Task 6 completed. Predictions saved.")
        
        return predictions, eval_metrics
    
    def generate_report(self):
        """Generate comprehensive student project report"""
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*80)
        
        report_path = self.results_dir / 'PROJECT_REPORT.md'
        
        with open(report_path, 'w') as f:
            f.write("# Drug Category Prediction Based on Gene Expression in Brain\n\n")
            f.write("## Student Project Report\n\n")
            f.write("---\n\n")
            
            # Introduction
            f.write("## 1. Introduction\n\n")
            f.write("This project implements a Neuro-Symbolic (NeSy) system for predicting drug ")
            f.write("therapeutic categories based on their effects on gene expression in brain cell cluster types.\n\n")
            
            # System Architecture
            f.write("## 2. System Architecture\n\n")
            f.write("The system follows a modular design pattern with 6 main tasks:\n\n")
            
            # Task descriptions
            for i, task_desc in enumerate([
                ("Gene Expression by Brain Cell Cluster Type", "Analyze scRNA-seq data to understand gene expression patterns across different brain cell cluster types"),
                ("Extract Drug Target Genes", "Combine DrugBank and DRUGseqr.gmt to extract comprehensive drug-gene interactions"),
                ("Fetch ATC Hierarchy", "Extract therapeutic classification from DrugBank ATC codes"),
                ("Integrate Gene Signatures", "Create feature embeddings combining brain cell cluster type expression, drug targets, and gene signatures"),
                ("CNS vs Non-CNS Classification (Stage 1)", "Train binary classifier to distinguish CNS-active from Non-CNS drugs"),
                ("Drug Category Classification (Stage 2)", "Learn patterns to predict drug therapeutic categories based on gene effects"),
                ("Prediction and Evaluation", "Generate top-k predictions with probabilities and comprehensive evaluation metrics")
            ], 1):
                f.write(f"### Task {i}: {task_desc[0]}\n")
                f.write(f"- **Description**: {task_desc[1]}\n")
                f.write(f"- **Output File**: `task{i}_*.csv` or `task{i}_*.npy`\n\n")
            
            # Dataset Information
            f.write("## 3. Datasets Used\n\n")
            f.write("1. **DrugBank XML**: Comprehensive drug database with targets, ATC codes\n")
            f.write("2. **DRUGseqr.gmt**: Gene expression signatures from drug perturbations\n")
            f.write("3. **scRNA Brain Data**: Single-cell RNA-seq from brain cell cluster types\n\n")
            
            # Results
            f.write("## 4. Results and Evaluation\n\n")
            
            if self.cns_metrics:
                f.write("### CNS Classification (Stage 1) Performance\n\n")
                for metric, value in self.cns_metrics.items():
                    if isinstance(value, (int, float, np.number)):
                        f.write(f"- **{metric}**: {value:.4f}\n")
                    else:
                        f.write(f"- **{metric}**: {value}\n")
                f.write("\n")
            
            if 'training' in self.metrics:
                f.write("### Drug Category Classification (Stage 2) Performance\n\n")
                for metric, value in self.metrics['training'].items():
                    f.write(f"- **{metric}**: {value}\n")
                f.write("\n")
            
            if 'evaluation' in self.metrics:
                f.write("### Test Performance\n\n")
                for metric, value in self.metrics['evaluation'].items():
                    if isinstance(value, (int, float, np.number)):
                        f.write(f"- **{metric}**: {value:.4f}\n")
                    else:
                        f.write(f"- **{metric}**: {value}\n")
                f.write("\n")
            
            # Explainability
            f.write("## 5. Explainability\n\n")
            f.write("Each task is independently implemented and can be analyzed:\n")
            f.write("- Task outputs are saved as CSV/NPY files for inspection\n")
            f.write("- Feature importance can be extracted from the classification model\n")
            f.write("- Top-k predictions provide confidence scores for interpretability\n\n")
            
            # Conclusion
            f.write("## 6. Conclusion\n\n")
            f.write("This NeSy system successfully integrates multiple data sources to predict ")
            f.write("drug therapeutic categories based on gene expression effects in brain cell cluster types. ")
            f.write("The modular architecture ensures each step is explainable and verifiable.\n\n")
            
            # Files Generated
            f.write("## 7. Generated Files\n\n")
            f.write("```\n")
            f.write("results/\n")
            for file in sorted(self.results_dir.glob('task*.*')):
                f.write(f"  - {file.name}\n")
            f.write("  - PROJECT_REPORT.md\n")
            f.write("  - EVALUATION_METRICS.json\n")
            f.write("```\n")
        
        print(f"✓ Report generated: {report_path}")
        
        # Save metrics as JSON
        import json
        metrics_path = self.results_dir / 'EVALUATION_METRICS.json'
        with open(metrics_path, 'w') as f:
            # Convert numpy types to Python types
            metrics_clean = {}
            for key, value in self.metrics.items():
                if isinstance(value, dict):
                    metrics_clean[key] = {}
                    for k, v in value.items():
                        if isinstance(v, (np.integer, np.floating)):
                            metrics_clean[key][k] = float(v)
                        elif isinstance(v, np.ndarray):
                            metrics_clean[key][k] = v.tolist()
                        else:
                            metrics_clean[key][k] = v
                else:
                    metrics_clean[key] = value
            json.dump(metrics_clean, f, indent=2)
        print(f"✓ Metrics saved: {metrics_path}")
    
    def run(self):
        """Execute the complete NeSy pipeline"""
        print("\n" + "="*80)
        print("NEURO-SYMBOLIC DRUG CATEGORY PREDICTION SYSTEM")
        print("Following gene-correlation.json Design Pattern")
        print("="*80)
        
        # Step 0: Validate datasets
        if not self.validate_datasets():
            print("\n✗ Dataset validation failed. Aborting pipeline.")
            return
        
        # Task 1: Brain gene expression
        self.task1_brain_regions()
        
        # Task 2: Drug targets
        self.task2_drug_targets()
        
        # Task 3: ATC hierarchy
        self.task3_atc_categories()
        
        # Task 4: Integration
        X, valid_indices = self.task4_integration()
        
        # Task 5a: CNS Classification (Stage 1)
        self.task5a_cns_classification(X, valid_indices)
        
        # Prepare labels from ATC hierarchy for drug category classification
        print("\n  Preparing labels from ATC hierarchy for drug category prediction...")
        drug_to_atc = self.atc_hierarchy.groupby('drug_id')['primary_category'].first()
        
        labels = []
        valid_feature_indices = []
        for i, idx in enumerate(valid_indices):
            drug_id = self.drug_targets.loc[idx, 'drug_id']
            if drug_id in drug_to_atc.index:
                labels.append(drug_to_atc[drug_id])
                valid_feature_indices.append(i)
        
        y = np.array(labels)
        X_filtered = X[valid_feature_indices]
        
        print(f"    Total samples with labels: {len(y)}")
        print(f"    Unique categories: {len(np.unique(y))}")
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_filtered, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"    Train set: {len(X_train)}, Test set: {len(X_test)}")
        
        # Task 5: Train drug category classification model (Stage 2)
        self.task5b_train_classifier_model(X_train, y_train)
        
        # Task 6: Predict and evaluate
        self.task6_predict(self.classifier_model, X_test, y_test, top_k=self.config.get('top_k', 5))
        
        # Generate report
        self.generate_report()
        
        print("\n" + "="*80)
        print("✓ NESY SYSTEM PIPELINE COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"\nAll results saved to: {self.results_dir}")
        print("Check PROJECT_REPORT.md for detailed analysis")


def main():
    """Main entry point"""
    
    # Configuration
    config = {
        'drugbank_xml': r"c:\Users\user\OneDrive\Desktop\My Files\LUH\Semester 5\DrugCategoryPrediction\drugbank_all_full_database.xml\full_database.xml",
        'cluster_type_tsv': r"c:\Users\user\OneDrive\Desktop\My Files\LUH\Semester 5\DrugCategoryPrediction\dataset\rna_single_nuclei_cluster_type.tsv\rna_single_nuclei_cluster_type.tsv",
        'drugseqr_gmt': r"c:\Users\user\OneDrive\Desktop\My Files\LUH\Semester 5\DrugCategoryPrediction\dataset\DRUGseqr.gmt",
        'results_dir': 'results',
        'top_k': 5
    }
    
    # Create and run system
    system = NeSyDrugPredictionSystem(config)
    system.run()


if __name__ == "__main__":
    main()