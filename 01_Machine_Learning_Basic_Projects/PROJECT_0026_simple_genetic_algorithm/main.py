"""
Project 026: Simple genetic algorithm

Overview:
  This project builds a production-minded simple genetic algorithm workflow inside the broader area of foundational machine learning. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import random

# Genetic Algorithm to solve the 0/1 Knapsack Problem
weights = [10, 20, 30, 40, 50]
values = [20, 30, 66, 40, 60]
capacity = 100

def fitness(chromosome):
    total_w = sum(w for w, bit in zip(weights, chromosome) if bit == 1)
    total_v = sum(v for v, bit in zip(values, chromosome) if bit == 1)
    return total_v if total_w <= capacity else 0

random.seed(42)
pop_size = 20
population = [[random.choice([0, 1]) for _ in range(len(weights))] for _ in range(pop_size)]

for generation in range(50):
    population = sorted(population, key=fitness, reverse=True)
    next_gen = population[:4]  # Elitism
    while len(next_gen) < pop_size:
        p1, p2 = random.sample(population[:10], 2)
        idx = random.randint(1, len(weights) - 1)
        child = p1[:idx] + p2[idx:]
        if random.random() < 0.15:  # Mutation
            mut_idx = random.randint(0, len(weights) - 1)
            child[mut_idx] = 1 - child[mut_idx]
        next_gen.append(child)
    population = next_gen

best = max(population, key=fitness)
print("=== Simple Genetic Algorithm (Knapsack Solution) ===")
print("Optimal Selection Vector:", best)
print(f"Max Value Achieved: {fitness(best)} (Total Weight: {sum(w for w, bit in zip(weights, best) if bit == 1)}/{capacity})")
