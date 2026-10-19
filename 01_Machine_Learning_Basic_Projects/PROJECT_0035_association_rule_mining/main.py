"""
Project 035: Association rule mining

Overview:
  This project builds a production-minded association rule mining workflow inside the broader area of foundational machine learning. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

# Association Rule Mining from Scratch
transactions = [
    {'milk', 'bread'},
    {'bread', 'butter'},
    {'milk', 'bread', 'butter'},
    {'beer', 'diapers'},
    {'milk', 'diapers'},
    {'milk', 'bread', 'butter'}
]
N = len(transactions)

def get_support(itemset):
    return sum(1 for t in transactions if itemset.issubset(t)) / N

rule_antecedent = {'bread'}
rule_consequent = {'butter'}

supp_A = get_support(rule_antecedent)
supp_B = get_support(rule_consequent)
supp_AB = get_support(rule_antecedent.union(rule_consequent))
confidence = supp_AB / supp_A
lift = confidence / supp_B

print("=== Association Rule Mining ===")
print("Rule: {bread} => {butter}")
print(f"Support(bread): {supp_A:.2f}")
print(f"Confidence: {confidence:.2f}")
print(f"Lift: {lift:.2f} (Lift > 1 indicates positive association)")
