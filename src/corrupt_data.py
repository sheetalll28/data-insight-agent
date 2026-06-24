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

def inject_mislabels(data, mislabel_prob=0.05):
    """
    Introduces typos or alternate naming conventions to categorical columns.
    This mimics real-world human data entry errors.
    """
    corrupted_data = []
    ground_truth_log = []
    
    # Let's define some realistic shorthand and typos for our specific dataset
    typo_map = {
        'Gender': {
            'Male': ['M', 'male', 'man', 'Mle'],
            'Female': ['F', 'female', 'woman', 'Femle']
        },
        'Diagnosis': {
            'Major Depressive Disorder': ['Depression', 'MDD', 'Major Depression'],
            'Generalized Anxiety Disorder': ['Anxiety', 'GAD', 'Gen. Anxiety'],
            'Bipolar Disorder': ['Bipolar', 'BD', 'Bipolar Dis.'],
            'Schizophrenia': ['Schizo', 'schizophrenia']
        }
    }

    target_columns = ['Gender', 'Diagnosis']

    for i, row in enumerate(data):
        new_row = row.copy()
        for col in target_columns:
            if random.random() < mislabel_prob:
                true_val = row[col]
                # If we have predefined typos for this exact category, pick one randomly
                if true_val in typo_map[col]:
                    bad_val = random.choice(typo_map[col][true_val])
                    
                    # Log the truth before we overwrite it
                    ground_truth_log.append({
                        'row_index': i,
                        'patient_id': row['Patient ID'],
                        'column': col,
                        'original_value': true_val,
                        'corruption_type': 'mislabel',
                        'corrupted_value': bad_val
                    })
                    
                    new_row[col] = bad_val
        corrupted_data.append(new_row)
        
    return corrupted_data, ground_truth_log

def inject_outliers(data, outlier_prob=0.05):
    """
    Injects numerically implausible values into numeric columns.
    This simulates sensor errors or human "fat-finger" data entry.
    """
    corrupted_data = []
    ground_truth_log = []
    
    # We will target Age and Adherence
    target_columns = ['Age', 'Adherence to Treatment (%)']
    
    for i, row in enumerate(data):
        new_row = row.copy()
        for col in target_columns:
            # We must make sure the cell isn't already blank from our 'missing values' step!
            if row[col] != '' and random.random() < outlier_prob:
                true_val = row[col]
                
                # Generate an absurd outlier based on the column
                if col == 'Age':
                    # Age is normally 18-100. Let's make it highly unrealistic
                    bad_val = str(random.choice([-15, 0, 150, 999]))
                elif col == 'Adherence to Treatment (%)':
                    # Percentage should be 0-100
                    bad_val = str(random.choice([-50, 200, 9999]))
                else:
                    bad_val = "999"
                    
                # Log the truth
                ground_truth_log.append({
                    'row_index': i,
                    'patient_id': row['Patient ID'],
                    'column': col,
                    'original_value': true_val,
                    'corruption_type': 'outlier',
                    'corrupted_value': bad_val
                })
                
                new_row[col] = bad_val
        corrupted_data.append(new_row)
        
    return corrupted_data, ground_truth_log

def inject_duplicates(data, duplicate_prob=0.05):
    """
    Simulates a patient being entered into the system twice under two different IDs.
    We keep their demographics identical but slightly tweak a clinical field.
    This creates a realistic "Entity Resolution" challenge for our future algorithm.
    """
    corrupted_data = list(data) # Make a copy of the list
    ground_truth_log = []
    
    # Find the highest existing Patient ID so we can assign new ones
    current_max_id = max([int(row['Patient ID']) for row in data if str(row['Patient ID']).isdigit()])
    next_new_id = current_max_id + 1
    
    new_duplicates = []
    
    for i, row in enumerate(data):
        if random.random() < duplicate_prob:
            dup_row = row.copy()
            
            # 1. Give them a brand new Patient ID (so it's not a trivial exact-match deduplication)
            dup_row['Patient ID'] = str(next_new_id)
            
            # 2. Tweak a value slightly to simulate a typo during the second entry
            if dup_row['Symptom Severity (1-10)'].isdigit():
                val = int(dup_row['Symptom Severity (1-10)'])
                # Shift by +1 or -1, keeping it within bounds 1-10
                new_val = max(1, min(10, val + random.choice([-1, 1])))
                dup_row['Symptom Severity (1-10)'] = str(new_val)
                
            new_duplicates.append(dup_row)
            
            # 3. Log the truth: "This new ID is actually a clone of the old ID"
            ground_truth_log.append({
                'row_index': 'unknown_due_to_shuffle',
                'patient_id': str(next_new_id),
                'column': 'Entire Row',
                'original_value': str(row['Patient ID']), # The ID it was cloned from
                'corruption_type': 'duplicate',
                'corrupted_value': 'N/A'
            })
            
            next_new_id += 1
            
    # Add the duplicates to our dataset
    corrupted_data.extend(new_duplicates)
    
    # Shuffle the dataset so the duplicates are hidden randomly throughout the file!
    random.shuffle(corrupted_data)
    
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
    messy_data, missing_log = inject_missing_values(raw_data, missing_prob=0.05)
    
    # 3. Inject mislabels (stacking it on top of the already-messy data)
    print("Injecting mislabels...")
    messy_data, mislabel_log = inject_mislabels(messy_data, mislabel_prob=0.05)
    
    # 4. Inject outliers
    print("Injecting outliers...")
    messy_data, outlier_log = inject_outliers(messy_data, outlier_prob=0.05)
    
    # 5. Inject duplicate rows
    print("Injecting duplicate rows...")
    messy_data, duplicate_log = inject_duplicates(messy_data, duplicate_prob=0.05)
    
    # Combine all answer keys
    truth_log = missing_log + mislabel_log + outlier_log + duplicate_log
    
    # 6. Save the messy dataset
    print("Saving messy_data.csv...")
    with open(messy_data_path, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(messy_data)
        
    # 4. Save the ground truth answer key
    print("Saving ground_truth.json...")
    with open(ground_truth_path, mode='w', encoding='utf-8') as f:
        json.dump(truth_log, f, indent=4)
        
    print(f"Done! Created messy_data.csv. Logged {len(missing_log)} missing values, {len(mislabel_log)} mislabels, {len(outlier_log)} outliers, and {len(duplicate_log)} duplicates.")

if __name__ == "__main__":
    main()
