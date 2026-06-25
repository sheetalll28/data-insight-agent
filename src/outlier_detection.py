import numpy as np
import csv
import json
import os

def calculate_iqr_bounds(data_array):
    """
    Calculates the Interquartile Range (IQR) entirely from scratch.
    It finds the 25th percentile (Q1) and 75th percentile (Q3), 
    then calculates the safe mathematical boundaries for normal data.
    """
    # 1. Sort the data from lowest to highest
    sorted_data = np.sort(data_array)
    n = len(sorted_data)
    
    if n == 0:
        return 0, 0
        
    # 2. Find Q1 (25% mark) and Q3 (75% mark)
    q1_idx = int(n * 0.25)
    q3_idx = int(n * 0.75)
    
    Q1 = sorted_data[q1_idx]
    Q3 = sorted_data[q3_idx]
    
    # 3. Calculate IQR (The middle 50% of the data)
    IQR = Q3 - Q1
    
    # 4. Calculate Lower and Upper Bounds
    # The 1.5 multiplier is the statistical industry standard created by John Tukey
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    return lower_bound, upper_bound

def evaluate_performance(predicted_outliers, truth_log_path):
    """
    Calculates Precision, Recall, and F1-Score from scratch.
    """
    with open(truth_log_path, 'r', encoding='utf-8') as f:
        truth_log = json.load(f)
        
    true_outliers = set()
    for entry in truth_log:
        if entry['corruption_type'] == 'outlier':
            # We track outliers by (Patient ID, Column) so we know exactly where it happened
            pair = (str(entry['patient_id']), entry['column'])
            true_outliers.add(pair)
            
    # Calculate True Positives, False Positives, False Negatives
    true_positives = len(predicted_outliers.intersection(true_outliers))
    false_positives = len(predicted_outliers - true_outliers)
    false_negatives = len(true_outliers - predicted_outliers)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print("\n" + "="*40)
    print("--- ML PERFORMANCE EVALUATION ---")
    print("="*40)
    print(f"True Outliers Injected : {len(true_outliers)}")
    print(f"Total Outliers Flagged : {len(predicted_outliers)}")
    print(f"Correct Flags (TP)     : {true_positives}")
    print(f"False Alarms (FP)      : {false_positives}")
    print(f"Missed Outliers (FN)   : {false_negatives}")
    print("-" * 40)
    print(f"Precision : {precision:.3f} (When the Agent guesses, how often is it right?)")
    print(f"Recall    : {recall:.3f} (Out of all real outliers, how many did the Agent find?)")
    print(f"F1-Score  : {f1:.3f} (Harmonic mean of both)")
    print("="*40 + "\n")

def main():
    print("Starting Outlier Detection Agent...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    messy_path = os.path.join(base_dir, 'data', 'processed', 'messy_data.csv')
    
    with open(messy_path, mode='r', encoding='utf-8') as f:
        data = list(csv.DictReader(f))
        
    numeric_cols = [
        'Age', 'Symptom Severity (1-10)', 'Mood Score (1-10)', 
        'Sleep Quality (1-10)', 'Physical Activity (hrs/week)', 
        'Treatment Duration (weeks)', 'Stress Level (1-10)', 
        'Treatment Progress (1-10)', 'Adherence to Treatment (%)'
    ]
    
    predicted_outliers = set()
    
    # 1. Scan column by column
    for col in numeric_cols:
        # Extract valid numbers for this column to build the math bounds
        valid_numbers = []
        for row in data:
            # Check if it's a number (allowing for negative signs and decimals)
            if row[col].lstrip('-').replace('.', '', 1).isdigit():
                valid_numbers.append(float(row[col]))
                
        if not valid_numbers:
            continue
            
        # 2. Calculate the safe mathematical bounds for this specific column
        lower, upper = calculate_iqr_bounds(np.array(valid_numbers))
        
        # 3. Scan the column again and flag the specific patients who violate the bounds
        for row in data:
            if row[col].lstrip('-').replace('.', '', 1).isdigit():
                val = float(row[col])
                if val < lower or val > upper:
                    pair = (str(row['Patient ID']), col)
                    predicted_outliers.add(pair)
                    
    print(f"Scan complete. Agent flagged {len(predicted_outliers)} cells as impossible anomalies.")
                    
    # 4. Evaluate against ground truth
    truth_path = os.path.join(base_dir, 'data', 'processed', 'ground_truth.json')
    evaluate_performance(predicted_outliers, truth_path)

if __name__ == "__main__":
    main()
