import csv
import sys
import os

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
    
    # 1. Exact Match for Demographics (Heavy Weight)
    if row1['Gender'] != '' and row2['Gender'] != '':
        weight = 3.0
        weights_total += weight
        if row1['Gender'] == row2['Gender']:
            score += weight
            
    if row1['Age'] != '' and row2['Age'] != '':
        weight = 3.0
        weights_total += weight
        if row1['Age'] == row2['Age']:
            score += weight
            
    # 2. Fuzzy Match for Text Fields using our Levenshtein Math (Medium Weight)
    text_fields = ['Diagnosis', 'Therapy Type', 'Medication']
    for field in text_fields:
        if row1[field] != '' and row2[field] != '':
            weight = 2.0
            weights_total += weight
            # Here we use our from-scratch Levenshtein distance!
            sim = string_similarity(row1[field], row2[field])
            score += (sim * weight)
            
    # 3. Numeric Proximity for Clinical Scores (Low Weight)
    numeric_fields = ['Symptom Severity (1-10)', 'Stress Level (1-10)']
    for field in numeric_fields:
        # Check if they are valid numbers (ignoring missing or weird outliers)
        if row1[field].lstrip('-').isdigit() and row2[field].lstrip('-').isdigit():
            weight = 1.0
            weights_total += weight
            v1, v2 = int(row1[field]), int(row2[field])
            diff = abs(v1 - v2)
            # Convert difference into a similarity (0 diff = 1.0 sim)
            sim = max(0.0, 1.0 - (diff / 10.0))
            score += (sim * weight)
            
    if weights_total == 0:
        return 0.0
        
    return score / weights_total

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

if __name__ == "__main__":
    main()
