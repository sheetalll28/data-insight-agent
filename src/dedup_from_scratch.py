import csv
import sys
import os
import json

# Ensure we can import our Levenshtein script
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)
from levenshtein import string_similarity

class UnionFind:
    """
    A Disjoint-Set (Union-Find) data structure from scratch.
    It tracks a set of elements partitioned into a number of disjoint (non-overlapping) subsets.
    This is critical for clustering duplicate records: if A matches B, and B matches C, 
    this structure mathematically ensures A, B, and C are grouped together.
    """
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, i):
        """Finds the root representative of the cluster containing 'i', with path compression."""
        # If 'i' is not in the system yet, it is its own root
        if i not in self.parent:
            self.parent[i] = i
            self.rank[i] = 0
            return i
            
        # Path compression: make the tree flat so future lookups are O(1)
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        """Unites the clusters containing 'i' and 'j', using union by rank."""
        root_i = self.find(i)
        root_j = self.find(j)

        # If they aren't already in the same cluster, merge them!
        if root_i != root_j:
            # Union by rank keeps the tree shallow and fast
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1

    def get_clusters(self):
        """Returns a list of all clusters found."""
        clusters = {}
        for item in self.parent:
            root = self.find(item)
            if root not in clusters:
                clusters[root] = set()
            clusters[root].add(item)
            
        # Only return clusters that actually have more than 1 item (actual duplicates)
        return [c for c in clusters.values() if len(c) > 1]

def calculate_similarity(row1, row2):
    """
    Compares two patient records and returns a confidence score (0.0 to 1.0)
    that they are the same person.
    """
    score = 0.0
    weights_total = 0.0
    
    # 1. Demographics (Medium Weight)
    if row1['Gender'] != '' and row2['Gender'] != '':
        weights_total += 2.0
        if row1['Gender'] == row2['Gender']: score += 2.0
            
    if row1['Age'] != '' and row2['Age'] != '':
        weights_total += 2.0
        if row1['Age'] == row2['Age']: score += 2.0
        
    # 2. Temporal Anchor (MASSIVE Weight)
    # If two people have the exact same demographics AND started treatment on the exact same day,
    # it's incredibly likely they are our duplicate rows.
    if row1['Treatment Start Date'] != '' and row2['Treatment Start Date'] != '':
        weights_total += 6.0
        if row1['Treatment Start Date'] == row2['Treatment Start Date']: score += 6.0
            
    # 3. Fuzzy Match for all Text Fields using Levenshtein (Standard Weight)
    text_fields = ['Diagnosis', 'Therapy Type', 'Medication', 'Outcome', 'AI-Detected Emotional State']
    for field in text_fields:
        if row1[field] != '' and row2[field] != '':
            weights_total += 1.0
            sim = string_similarity(row1[field], row2[field])
            score += sim
            
    # 4. Numeric Proximity for all Clinical Scores (Low Weight)
    numeric_fields = ['Symptom Severity (1-10)', 'Stress Level (1-10)', 'Mood Score (1-10)', 
                      'Sleep Quality (1-10)', 'Physical Activity (hrs/week)', 
                      'Treatment Duration (weeks)', 'Treatment Progress (1-10)', 'Adherence to Treatment (%)']
    for field in numeric_fields:
        if row1[field].lstrip('-').isdigit() and row2[field].lstrip('-').isdigit():
            weights_total += 0.5
            v1, v2 = int(row1[field]), int(row2[field])
            diff = abs(v1 - v2)
            
            # Decay the similarity based on how far apart the numbers are
            if field == 'Adherence to Treatment (%)':
                sim = max(0.0, 1.0 - (diff / 20.0)) # 20% diff = 0 sim
            else:
                sim = max(0.0, 1.0 - (diff / 5.0))  # 5 point diff = 0 sim
                
            score += (sim * 0.5)
            
    if weights_total == 0:
        return 0.0
        
    return score / weights_total

def evaluate_performance(predicted_clusters, truth_log_path):
    """
    Calculates Precision, Recall, and F1-Score entirely from scratch.
    """
    # 1. Extract the True Pairs from our answer key
    with open(truth_log_path, 'r', encoding='utf-8') as f:
        truth_log = json.load(f)
        
    true_pairs = set()
    for entry in truth_log:
        if entry['corruption_type'] == 'duplicate':
            # We sort them so (Patient A, Patient B) is treated identically to (Patient B, Patient A)
            pair = tuple(sorted([str(entry['patient_id']), str(entry['original_value'])]))
            true_pairs.add(pair)
            
    # 2. Extract the Predicted Pairs from our clusters
    predicted_pairs = set()
    for cluster in predicted_clusters:
        cluster_list = list(cluster)
        # Generate all pairwise combinations within the cluster
        for i in range(len(cluster_list)):
            for j in range(i + 1, len(cluster_list)):
                pair = tuple(sorted([str(cluster_list[i]), str(cluster_list[j])]))
                predicted_pairs.add(pair)
                
    # 3. Calculate True Positives, False Positives, False Negatives
    true_positives = len(predicted_pairs.intersection(true_pairs))
    false_positives = len(predicted_pairs - true_pairs)
    false_negatives = len(true_pairs - predicted_pairs)
    
    # 4. Calculate Precision, Recall, F1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print("\n" + "="*40)
    print("--- ML PERFORMANCE EVALUATION ---")
    print("="*40)
    print(f"True Duplicates Hidden in Data : {len(true_pairs)}")
    print(f"Total Pairs Guessed by Agent   : {len(predicted_pairs)}")
    print(f"Correct Guesses (True Positive): {true_positives}")
    print(f"Wrong Guesses (False Positive) : {false_positives}")
    print(f"Missed Duplicates (False Neg.) : {false_negatives}")
    print("-" * 40)
    print(f"Precision : {precision:.3f} (When the Agent guesses, how often is it right?)")
    print(f"Recall    : {recall:.3f} (Out of all real duplicates, how many did the Agent find?)")
    print(f"F1-Score  : {f1:.3f} (Harmonic mean of both)")
    print("="*40 + "\n")

def main():
    print("Starting Entity Resolution Agent...")
    
    # 1. Load the messy data
    base_dir = os.path.dirname(script_dir)
    messy_path = os.path.join(base_dir, 'data', 'processed', 'messy_data.csv')
    
    with open(messy_path, mode='r', encoding='utf-8') as f:
        data = list(csv.DictReader(f))
        
    print(f"Loaded {len(data)} records.")
    
    # 2. Initialize Union-Find
    uf = UnionFind()
    
    # Register all patients as their own unique root first
    for row in data:
        uf.find(row['Patient ID'])
        
    # 3. Pairwise Comparison O(N^2)
    # We compare every row to every OTHER row exactly once
    print("Comparing all pairs... (this might take a few seconds)")
    comparisons_made = 0
    matches_found = 0
    
    THRESHOLD = 0.85 # We must be 85% confident to merge them
    
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            comparisons_made += 1
            sim_score = calculate_similarity(data[i], data[j])
            
            if sim_score >= THRESHOLD:
                uf.union(data[i]['Patient ID'], data[j]['Patient ID'])
                matches_found += 1
                
    # 4. Get the results
    clusters = uf.get_clusters()
    
    print(f"\nFinished {comparisons_made} comparisons.")
    print(f"The Agent decided to merge {matches_found} pairs!")
    print(f"Total Duplicate Clusters Found: {len(clusters)}")
    
    # Print the first few clusters as an example
    for idx, cluster in enumerate(clusters[:5]):
        print(f"Cluster {idx+1}: Patient IDs {cluster}")
        
    # 5. Evaluate our Performance
    truth_path = os.path.join(base_dir, 'data', 'processed', 'ground_truth.json')
    evaluate_performance(clusters, truth_path)

if __name__ == "__main__":
    main()
