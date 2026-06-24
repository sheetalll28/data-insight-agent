import csv
import random
import json
import os

def load_data(filepath):
    """Reads the CSV into a list of dictionaries."""
    with open(filepath, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames

def inject_missing_values(data, missing_prob=0.05):
    """
    Randomly removes values from certain columns to simulate missing data.
    We log the original value so we have a 'ground truth' to check against later.
    """
    corrupted_data = []
    ground_truth_log = []
    
    # Let's pick a few numerical columns that commonly have missing data in the real world
    target_columns = ['Age', 'Sleep Quality (1-10)', 'Stress Level (1-10)']

    for i, row in enumerate(data):
        new_row = row.copy() # Make a copy so we don't modify the original yet
        for col in target_columns:
            # random.random() gives a float between 0.0 and 1.0
            if random.random() < missing_prob:
                # 1. Log the truth before we destroy it
                ground_truth_log.append({
                    'row_index': i,
                    'patient_id': row['Patient ID'],
                    'column': col,
                    'original_value': row[col],
                    'corruption_type': 'missing'
                })
                # 2. Destroy the data (simulate a blank cell in a CSV)
                new_row[col] = ''
                
        corrupted_data.append(new_row)
        
    return corrupted_data, ground_truth_log

def main():
    random.seed(42) # For reproducibility while we build
    
    # Calculate absolute paths so the script works no matter where you run it from!
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir) # Go up one level to data-insight-agent
    raw_data_path = os.path.join(base_dir, 'data', 'raw', 'mental_health_diagnosis_treatment_.csv')
    messy_data_path = os.path.join(base_dir, 'data', 'processed', 'messy_data.csv')
    ground_truth_path = os.path.join(base_dir, 'data', 'processed', 'ground_truth.json')
    
    # 1. Load the clean dataset
    print("Loading raw data...")
    raw_data, fieldnames = load_data(raw_data_path)
    
    # 2. Inject missing values
    print("Injecting missing values...")
    messy_data, truth_log = inject_missing_values(raw_data, missing_prob=0.05)
    
    # 3. Save the messy dataset
    print("Saving messy_data.csv...")
    with open(messy_data_path, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(messy_data)
        
    # 4. Save the ground truth answer key
    print("Saving ground_truth.json...")
    with open(ground_truth_path, mode='w', encoding='utf-8') as f:
        json.dump(truth_log, f, indent=4)
        
    print(f"Done! Created messy_data.csv and logged {len(truth_log)} missing values.")

if __name__ == "__main__":
    main()
