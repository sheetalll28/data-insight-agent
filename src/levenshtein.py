def levenshtein_distance(s1, s2):
    """
    Calculates the minimum number of single-character edits (insertions, deletions, 
    or substitutions) required to change string s1 into string s2.
    
    Implemented from scratch using Dynamic Programming.
    """
    m, n = len(s1), len(s2)
    
    # Create a 2D grid (matrix) of size (m+1) x (n+1) initialized with zeros
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize the first row and column
    # If one string is empty, the distance is just the length of the other string
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
        
    # Fill in the grid
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # If the characters match, there is no cost to substitute!
            if s1[i-1] == s2[j-1]:
                cost = 0
            else:
                cost = 1 # We must pay 1 edit to substitute
            
            dp[i][j] = min(
                dp[i-1][j] + 1,      # Deletion (moving down in the grid)
                dp[i][j-1] + 1,      # Insertion (moving right in the grid)
                dp[i-1][j-1] + cost  # Substitution (moving diagonally)
            )
            
    # The bottom-right cell contains the final answer
    return dp[m][n]

def string_similarity(s1, s2):
    """
    Converts the Levenshtein distance into a similarity score between 0.0 and 1.0.
    1.0 means exact match, 0.0 means completely different.
    """
    if not s1 and not s2:
        return 1.0
    
    distance = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    
    return 1.0 - (distance / max_len)

if __name__ == "__main__":
    # Let's run some unit tests to prove our math is correct!
    print("Testing Levenshtein Distance from scratch...\n")
    
    test_cases = [
        ("CAT", "COAT", 1),       # 1 Insertion
        ("kitten", "sitting", 3), # 2 Substitutions, 1 Insertion
        ("Depression", "MDD", 10), # Completely different abbreviations
        ("Major Depression", "Major Depressive Disorder", 13),
        ("schizophrenia", "Schizo", 7)
    ]
    
    for s1, s2, expected in test_cases:
        dist = levenshtein_distance(s1, s2)
        sim = string_similarity(s1, s2)
        status = "PASSED" if dist == expected else f"FAILED (Got {dist})"
        print(f"[{status}] '{s1}' vs '{s2}'")
        print(f"  -> Distance: {dist} edits")
        print(f"  -> Similarity: {sim:.2f} ({(sim*100):.0f}% match)\n")
