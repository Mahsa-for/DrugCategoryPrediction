"""
Task 6: Prediction and Evaluation
Input: Trained model + Test data
Output: Top-k predicted therapeutic classes + probabilities
Process: Deduce (predict with confidence scores)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, top_k_accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns


def predict_top_k(model: Any, X_test: np.ndarray, top_k: int = 5) -> Tuple[List[Dict], np.ndarray]:
    """
    Generate top-k predictions with probabilities
    
    Returns:
        Tuple of (predictions_list, y_pred)
    """
    # Get probability predictions if available
    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba(X_test)
    else:
        # For models without predict_proba, use decision_function
        probas = model.decision_function(X_test)
    
    # Get class names
    classes = model.classes_
    
    # Generate predictions
    predictions = []
    y_pred = model.predict(X_test)
    
    for i in range(len(X_test)):
        # Get top-k predictions
        top_k_indices = np.argsort(probas[i])[::-1][:top_k]
        
        pred_dict = {
            'sample_idx': i,
            'top_1_class': classes[top_k_indices[0]],
            'top_1_prob': probas[i][top_k_indices[0]]
        }
        
        # Add top-k
        for k in range(min(top_k, len(top_k_indices))):
            pred_dict[f'top_{k+1}_class'] = classes[top_k_indices[k]]
            pred_dict[f'top_{k+1}_prob'] = probas[i][top_k_indices[k]]
        
        predictions.append(pred_dict)
    
    return predictions, y_pred


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> Dict[str, Any]:
    """
    Calculate comprehensive evaluation metrics
    """
    metrics = {}
    
    # Overall metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    metrics['per_class_report'] = report
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    metrics['confusion_matrix'] = cm
    metrics['class_names'] = class_names
    
    return metrics


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], output_path: Path):
    """
    Plot and save confusion matrix
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Drug Category Prediction')
    plt.ylabel('True Category')
    plt.xlabel('Predicted Category')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_per_class_metrics(report: Dict, output_path: Path):
    """
    Plot per-class performance metrics
    """
    # Extract metrics for each class
    classes = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
    
    precision = [report[c]['precision'] for c in classes]
    recall = [report[c]['recall'] for c in classes]
    f1 = [report[c]['f1-score'] for c in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width, precision, width, label='Precision', alpha=0.8)
    plt.bar(x, recall, width, label='Recall', alpha=0.8)
    plt.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    plt.xlabel('Drug Categories')
    plt.ylabel('Score')
    plt.title('Per-Class Performance Metrics')
    plt.xticks(x, classes, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def execute(model: Any,
            X_test: np.ndarray,
            y_test: np.ndarray,
            top_k: int,
            output_dir: Path) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Execute Task 6: Generate predictions and evaluate
    
    Args:
        model: Trained classification model
        X_test: Test features
        y_test: Test labels
        top_k: Number of top predictions to return
        output_dir: Directory to save outputs
    
    Returns:
        Tuple of (predictions, evaluation_metrics)
    """
    print("\n[Task 6] Generating predictions and evaluating model...")
    
    print(f"\n  Test set size: {len(X_test)}")
    print(f"  Generating top-{top_k} predictions...")
    
    # Generate predictions
    predictions, y_pred = predict_top_k(model, X_test, top_k)

    # Ensure y_pred and y_test are both string ATC codes for metrics
    if hasattr(model, 'classes_'):
        classes = np.array(model.classes_)
        if np.issubdtype(y_pred.dtype, np.integer):
            y_pred = classes[y_pred]
        if np.issubdtype(y_test.dtype, np.integer):
            y_test = classes[y_test]
        class_names = classes.tolist()
    else:
        class_names = sorted(list(set(y_test)))

    # Calculate metrics
    print(f"\n  Calculating evaluation metrics...")
    metrics = calculate_metrics(y_test, y_pred, class_names)
    
    # Print results
    print(f"\n  ═══════════════════════════════════════")
    print(f"  EVALUATION RESULTS")
    print(f"  ═══════════════════════════════════════")
    print(f"\n  Overall Performance:")
    print(f"    Accuracy:           {metrics['accuracy']:.4f}")
    print(f"    Precision (macro):  {metrics['precision_macro']:.4f}")
    print(f"    Recall (macro):     {metrics['recall_macro']:.4f}")
    print(f"    F1-Score (macro):   {metrics['f1_macro']:.4f}")
    print(f"    F1-Score (weighted):{metrics['f1_weighted']:.4f}")
    
    # Calculate top-k accuracy if possible
    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba(X_test)
        for k in [2, 3, 5]:
            if k <= len(class_names):
                top_k_acc = top_k_accuracy_score(y_test, probas, k=k, labels=class_names)
                print(f"    Top-{k} Accuracy:     {top_k_acc:.4f}")
                metrics[f'top_{k}_accuracy'] = top_k_acc
    
    # Per-class performance
    print(f"\n  Per-Class Performance:")
    report = metrics['per_class_report']
    for cls in class_names:
        if cls in report:
            print(f"    {cls}:")
            print(f"      Precision: {report[cls]['precision']:.4f}")
            print(f"      Recall:    {report[cls]['recall']:.4f}")
            print(f"      F1-Score:  {report[cls]['f1-score']:.4f}")
            print(f"      Support:   {int(report[cls]['support'])}")
    
    # Save detailed classification report
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(output_dir / 'task6_classification_report.csv')
    
    # Plot confusion matrix
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        class_names,
        output_dir / 'task6_confusion_matrix.png'
    )
    print(f"\n  ✓ Saved confusion matrix plot")
    
    # Plot per-class metrics
    plot_per_class_metrics(
        report,
        output_dir / 'task6_per_class_metrics.png'
    )
    print(f"  ✓ Saved per-class metrics plot")
    
    # Find best and worst predictions
    correct_mask = y_test == y_pred
    correct_count = correct_mask.sum()
    incorrect_count = (~correct_mask).sum()
    
    print(f"\n  Prediction Summary:")
    print(f"    Correct predictions:   {correct_count} ({correct_count/len(y_test)*100:.1f}%)")
    print(f"    Incorrect predictions: {incorrect_count} ({incorrect_count/len(y_test)*100:.1f}%)")
    
    # Save example predictions
    example_predictions = []
    for i in range(min(20, len(predictions))):
        pred = predictions[i]
        example_predictions.append({
            'sample_idx': i,
            'true_class': y_test[i],
            'predicted_class': pred['top_1_class'],
            'confidence': pred['top_1_prob'],
            'correct': y_test[i] == pred['top_1_class'],
            'top_2_class': pred.get('top_2_class', ''),
            'top_2_prob': pred.get('top_2_prob', 0.0)
        })
    
    example_df = pd.DataFrame(example_predictions)
    example_df.to_csv(output_dir / 'task6_example_predictions.csv', index=False)
    
    print(f"\n  ✓ Saved example predictions (first 20 samples)")
    
    return predictions, metrics


if __name__ == "__main__":
    print("Task 6 requires outputs from Tasks 1-5")
    print("Run the full pipeline to test this task")
