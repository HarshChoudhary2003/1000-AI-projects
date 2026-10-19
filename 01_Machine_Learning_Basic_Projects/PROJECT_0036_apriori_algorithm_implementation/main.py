"""
Project 036: Apriori algorithm implementation

Overview:
  This project builds a production-minded apriori algorithm implementation workflow inside the broader area of foundational machine learning. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

from itertools import combinations

# Apriori Algorithm Candidate Generation
transactions = [
    ['milk', 'bread', 'eggs'],
    ['bread', 'butter'],
    ['milk', 'bread', 'butter'],
    ['milk', 'eggs'],
    ['bread', 'butter', 'eggs']
]

min_support_count = 3
item_counts = {}
for t in transactions:
    for item in t:
        item_counts[item] = item_counts.get(item, 0) + 1

L1 = {frozenset([k]): v for k, v in item_counts.items() if v >= min_support_count}

print("=== Apriori Algorithm Implementation ===")
print(f"Frequent 1-Itemsets (min_count >= {min_support_count}):")
for itemset, count in L1.items():
    print(f"  {list(itemset)}: count={count}")
