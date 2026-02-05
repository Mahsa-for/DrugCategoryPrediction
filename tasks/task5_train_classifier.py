"""
Task 5: Train Drug Category Classification Model
Input: Integrated features + ATC categories
Output: Trained classification model
Process: Infer (learn drug category patterns)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, List
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score


def prepare_labels(drug_targets: pd.DataFrame, atc_hierarchy: pd.DataFrame, valid_indices: List[int]) -> Tuple[np.ndarray, List[str]]:
    """
    Prepare classification labels from ATC hierarchy
    
    Args:
        drug_targets: Drug target information
        atc_hierarchy: ATC hierarchical categories
        valid_indices: Indices of valid drugs
    
    Returns:
        Tuple of (labels, class_names)
    """
    # Map drug_ids to primary ATC categories
    drug_to_atc = atc_hierarchy.groupby('drug_id')['primary_category'].first()
    
    # Get labels for valid drugs
    labels = []
    for idx in valid_indices:
        drug_id = drug_targets.loc[idx, 'drug_id']
        if drug_id in drug_to_atc.index:
            labels.append(drug_to_atc[drug_id])
        else:
            labels.append('UNKNOWN')
    
    labels = np.array(labels)
    
    # Filter out UNKNOWN
    valid_mask = labels != 'UNKNOWN'
    labels = labels[valid_mask]
    
    class_names = sorted(list(set(labels)))
    
    return labels, class_names, valid_mask


def train_multiple_models(X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
    """
    Train and evaluate multiple classification models
    
    Returns:
        Dictionary with model performances
    """
    print("\n  Training multiple classification models...")
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', random_state=42, probability=True),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
    }
    
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"\n    Training {name}...")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
        
        # Fit on full training data
        model.fit(X_train, y_train)
        
        # Training accuracy
        train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        train_f1 = f1_score(y_train, train_pred, average='weighted')
        
        results[name] = {
            'model': model,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'train_accuracy': train_acc,
            'train_f1': train_f1
        }
        
        print(f"      CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"      Train Accuracy: {train_acc:.4f}")
        print(f"      Train F1: {train_f1:.4f}")
    
    # Select best model based on CV score
    best_model_name = max(results.keys(), key=lambda k: results[k]['cv_mean'])
    print(f"\n  ✓ Best model: {best_model_name} (CV: {results[best_model_name]['cv_mean']:.4f})")
    
    return results, best_model_name


def execute(X: np.ndarray,
            drug_targets: pd.DataFrame,
            atc_hierarchy: pd.DataFrame,
            valid_indices: List[int],
            output_dir: Path) -> Tuple[Any, Dict[str, Any]]:
    """
    Execute Task 5: Train classification model
    
    Args:
        X: Feature matrix
        drug_targets: Drug target information
        atc_hierarchy: ATC hierarchical categories
        valid_indices: Valid drug indices
        output_dir: Directory to save outputs
    
    Returns:
        Tuple of (best_model, metrics_dict)
    """
    print("\n[Task 5] Training drug category classification model...")
    
    # Prepare labels
    print("\n  Preparing classification labels...")
    y, class_names, valid_mask = prepare_labels(drug_targets, atc_hierarchy, valid_indices)
    
    # Filter X to match y
    X_filtered = X[valid_mask]
    
    print(f"    Total samples: {len(y)}")
    print(f"    Number of classes: {len(class_names)}")
    print(f"    Class names: {', '.join(class_names)}")
    
    # Class distribution
    print(f"\n  Class distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"    {cls}: {count} ({count/len(y)*100:.1f}%)")
    
    # Check for class imbalance
    min_samples = counts.min()
    max_samples = counts.max()
    imbalance_ratio = max_samples / min_samples
    print(f"\n  Class imbalance ratio: {imbalance_ratio:.2f}")
    
    if imbalance_ratio > 5:
        print(f"    ⚠ Warning: Significant class imbalance detected")
    
    # Train models
    results, best_model_name = train_multiple_models(X_filtered, y)
    
    best_model = results[best_model_name]['model']
    
    # Extract feature importance if available
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        importance_df = pd.DataFrame({
            'feature_idx': range(len(importances)),
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        importance_df.to_csv(output_dir / 'task5_feature_importances.csv', index=False)
        
        print(f"\n  Top 10 most important features:")
        for _, row in importance_df.head(10).iterrows():
            print(f"    Feature {int(row['feature_idx'])}: {row['importance']:.4f}")
    
    # Save model comparison
    comparison_data = []
    for name, result in results.items():
        comparison_data.append({
            'model': name,
            'cv_accuracy': result['cv_mean'],
            'cv_std': result['cv_std'],
            'train_accuracy': result['train_accuracy'],
            'train_f1': result['train_f1']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(output_dir / 'task5_model_comparison.csv', index=False)
    
    # Prepare metrics dictionary
    metrics = {
        'best_model': best_model_name,
        'cv_accuracy': results[best_model_name]['cv_mean'],
        'cv_std': results[best_model_name]['cv_std'],
        'train_accuracy': results[best_model_name]['train_accuracy'],
        'train_f1': results[best_model_name]['train_f1'],
        'num_classes': len(class_names),
        'num_samples': len(y),
        'class_names': class_names
    }
    
    return best_model, metrics


if __name__ == "__main__":
    print("Task 5 requires outputs from Tasks 1-4")
    print("Run the full pipeline to test this task")
