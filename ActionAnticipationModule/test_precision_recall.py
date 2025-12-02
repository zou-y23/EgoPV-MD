#!/usr/bin/env python3
import json
import sys
from pathlib import Path

def calculate_metrics(file1, file2):
    """
    Calculate Precision and Recall
    
    Comparison rules:
    - assistant_response same → predicted as "non-wrong action"
    - assistant_response different → predicted as "wrong action"
    
    Args:
        file1: Path to llama_generate1.json
        file2: Path to llama_generate2.json
    
    Returns:
        metrics: Dictionary containing various metrics
    """
    with open(file1, 'r', encoding='utf-8') as f:
        data1 = json.load(f)
    
    with open(file2, 'r', encoding='utf-8') as f:
        data2 = json.load(f)
    
    if len(data1) != len(data2):
        print(f"Warning: The two files have different number of entries!")
        return None
    
    TP = 0  # True Positive: predicted as wrong and actually wrong
    FP = 0  # False Positive: predicted as wrong but actually correct (false alarm)
    FN = 0  # False Negative: predicted as correct but actually wrong (missed detection)
    TN = 0  # True Negative: predicted as correct and actually correct
    
    for i in range(len(data1)):
        item1 = data1[i]
        item2 = data2[i]
        
        # Get assistant_response
        response1 = item1.get("assistant_response", "")
        response2 = item2.get("assistant_response", "")
        
        # Determine if predicted as wrong (response different)
        predicted_as_wrong = (response1 != response2)
        
        # Get ground truth label
        wrong = item1.get("wrong", False)
        if isinstance(wrong, str):
            wrong = wrong.lower() in ["true", "wrong"]
        elif not isinstance(wrong, bool):
            wrong = bool(wrong)
        
        # Count statistics
        if predicted_as_wrong and wrong:
            TP += 1  # Predicted as wrong, actually wrong
        elif predicted_as_wrong and not wrong:
            FP += 1  # Predicted as wrong, actually correct (false alarm)
        elif not predicted_as_wrong and wrong:
            FN += 1  # Predicted as correct, actually wrong (missed detection)
        else:
            TN += 1  # Predicted as correct, actually correct
    
    # Calculate metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0.0
    
    metrics = {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1_score,
        "Accuracy": accuracy,
        "Total": TP + FP + FN + TN
    }
    
    return metrics

def print_metrics(metrics):
    """Print metrics"""
    print("\n" + "="*70)
    print("Performance Metrics")
    print("="*70)
    print(f"Precision  = {metrics['TP']} / ({metrics['TP']} + {metrics['FP']}) = {metrics['Precision']:.4f} ({metrics['Precision']*100:.2f}%)")
    print(f"           (Proportion of predicted wrong samples that are actually wrong)")
    print(f"Recall     = {metrics['TP']} / ({metrics['TP']} + {metrics['FN']}) = {metrics['Recall']:.4f} ({metrics['Recall']*100:.2f}%)")
    print(f"           (Proportion of actually wrong samples that are detected)")
    print(f"F1-Score   = 2 * ({metrics['Precision']:.4f} * {metrics['Recall']:.4f}) / ({metrics['Precision']:.4f} + {metrics['Recall']:.4f}) = {metrics['F1_Score']:.4f}")


def main():
    """Main function"""
    file1 = "/root/autodl-tmp/LLaMA-Factory-main/data/AAM_generate.json"
    file2 = "/root/autodl-tmp/LLaMA-Factory-main/data/ARM_generate.json"
    
    if len(sys.argv) > 1:
        file1 = sys.argv[1]
    if len(sys.argv) > 2:
        file2 = sys.argv[2]
    
    if not Path(file1).exists():
        print(f"Error: File does not exist: {file1}")
        return
    
    if not Path(file2).exists():
        print(f"Error: File does not exist: {file2}")
        print(f"\nTip: Run the following command to generate file 2:")
        print(f"   python /root/autodl-tmp/generate_labels_for_test.py")
        return
    
    # Calculate metrics
    metrics = calculate_metrics(file1, file2)
    
    if metrics:
        # Print results
        print_metrics(metrics)

if __name__ == "__main__":
    main()
