import csv
import os
import sys
import numpy as np
import json

# Import our custom Neural Network
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)
from nn_from_scratch import NeuralNetwork

def prepare_data(data, target_col):
    """
    Prepares raw CSV data for a Neural Network.
    - Normalizes numeric columns (scales them to 0.0 - 1.0)
    - One-Hot Encodes categorical columns (e.g., Gender -> [1, 0] or [0, 1])
    """
    X = []
    y = []
    missing_indices = []
    
    # 1. Choose which columns we will use to predict the target
    numeric_inputs = ['Symptom Severity (1-10)', 'Stress Level (1-10)', 'Mood Score (1-10)']
    
    # 2. Collect all possible categories for One-Hot Encoding
    genders = list(set([row['Gender'] for row in data if row['Gender'] != '']))
    
    for i, row in enumerate(data):
        # Build the feature vector (the inputs) for this row
        features = []
        
        # A. Add Normalized Numeric Features
        for col in numeric_inputs:
            if row[col].lstrip('-').isdigit():
                val = float(row[col])
                
                # --- ML FIX: Outlier Clipping ---
                # Remember Phase 1 where we injected crazy outliers like 999 and -999?
                # We are feeding those into the Neural Network! Those massive numbers 
                # are completely destroying the gradients. We must clip them!
                val = max(1.0, min(10.0, val))
                
                # Normalize 1-10 score to 0.1 - 1.0 so the Neural Network can digest it easily
                features.append(val / 10.0)
            else:
                features.append(0.5) # Default to an average value if this input is also missing
                
        # B. Add One-Hot Encoded Categorical Features
        for gender in genders:
            if row['Gender'] == gender:
                features.append(1.0)
            else:
                features.append(0.0)
                
        # C. Check if our target (e.g., 'Age') is missing
        if row[target_col] == '':
            missing_indices.append(i)
            # We don't have the answer, so we append a placeholder
            y.append([0.0])
        else:
            y.append([float(row[target_col])])
            
        X.append(features)
        
    return np.array(X), np.array(y), missing_indices

def main():
    print("Initializing Data Preparation...")
    
    base_dir = os.path.dirname(script_dir)
    messy_path = os.path.join(base_dir, 'data', 'processed', 'messy_data.csv')
    
    with open(messy_path, mode='r', encoding='utf-8') as f:
        data = list(csv.DictReader(f))
        
    print(f"Loaded {len(data)} messy records.")
    
    # Prepare data to predict 'Age'
    X, y, missing_indices = prepare_data(data, 'Age')
    
    print(f"Prepared Input Data Shape: {X.shape} (Rows x Features)")
    print(f"Found {len(missing_indices)} patients missing their Age.")

    # 2. Split into Training Data (has Age) and Missing Data (needs Age)
    train_indices = [i for i in range(len(data)) if i not in missing_indices]
    
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_missing = X[missing_indices]
    
    # --- ML Trick: Target Normalization & Outlier Clipping ---
    # Neural Networks struggle when asked to output large numbers (like Age 55) directly. 
    # Even worse, we injected OUTLIERS into the Age column earlier (like Age 999 or -999)!
    # Training the network to predict a 999-year-old patient will completely destroy its math.
    # We must clip the target Age to a valid human range (18 to 100), and THEN squish it.
    y_train_clipped = np.clip(y_train, 18.0, 100.0)
    y_train_norm = y_train_clipped / 100.0
    
    # 3. Train the Neural Network
    print("\nTraining Neural Network from scratch...")
    input_size = X.shape[1]
    # We can also increase the learning rate slightly since our targets are nicely normalized
    nn = NeuralNetwork(input_size=input_size, hidden_size=8, output_size=1, learning_rate=0.1)
    
    epochs = 5000
    for epoch in range(epochs):
        # Forward Pass
        y_pred_norm = nn.forward(X_train)
        
        # Calculate Loss against NORMALIZED targets
        loss = nn.compute_loss(y_pred_norm, y_train_norm)
        
        # Backward Pass (Learn!)
        nn.backward(X_train, y_train_norm)
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}/{epochs} - Normalized Loss: {loss:.4f}")
            
    print(f"Final Normalized Loss: {loss:.4f}")
    
    # 4. Predict the missing values
    print("\nPredicting the 20 missing ages...")
    predictions_norm = nn.forward(X_missing)
    
    # --- Re-scale predictions back to real Ages ---
    predictions = predictions_norm * 100.0
    
    # 5. Evaluate against Ground Truth
    print("Evaluating predictions against secret answer key...")
    truth_path = os.path.join(base_dir, 'data', 'processed', 'ground_truth.json')
    with open(truth_path, 'r', encoding='utf-8') as f:
        truth_log = json.load(f)
        
    true_ages = []
    pred_ages = []
    
    for idx, missing_idx in enumerate(missing_indices):
        patient_id = data[missing_idx]['Patient ID']
        # Find this patient's true age in the secret log
        for entry in truth_log:
            if entry['patient_id'] == patient_id and entry['column'] == 'Age' and entry['corruption_type'] == 'missing':
                true_ages.append(float(entry['original_value']))
                pred_ages.append(predictions[idx][0])
                break
                
    true_ages = np.array(true_ages)
    pred_ages = np.array(pred_ages)
    
    # 6. Calculate Metrics (RMSE)
    def calculate_rmse(y_true, y_pred):
        return np.sqrt(np.mean(np.square(y_true - y_pred)))
        
    nn_rmse = calculate_rmse(true_ages, pred_ages)
    
    # Dumb baseline: just guess the average age of the training set
    mean_age = np.mean(y_train)
    baseline_preds = np.full_like(true_ages, mean_age)
    baseline_rmse = calculate_rmse(true_ages, baseline_preds)
    
    print("\n" + "="*40)
    print("--- IMPUTATION PERFORMANCE ---")
    print("="*40)
    print(f"Baseline RMSE (Guessing Average Age) : {baseline_rmse:.2f} years off")
    print(f"Neural Network RMSE                  : {nn_rmse:.2f} years off")
    print("-" * 40)
    if nn_rmse < baseline_rmse:
        print("WIN! Our Neural Network learned the hidden patterns and beat the baseline!")
    else:
        print("LOSS! The baseline was better. (The input features might not correlate strongly with Age).")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
