# ...existing code...

## Process Overview
1. Data Loading: The CSV file is read into memory and converted into a list of transactions.
2. Analysis Steps:
   - Apriori uses breadth-first search for frequent itemsets and computes support, confidence, and lift.
   - ECLAT focuses on vertical data structures and intersects transaction IDs for speed.
3. Output: The discovered rules are then converted to a DataFrame and visualized for quick insights.
