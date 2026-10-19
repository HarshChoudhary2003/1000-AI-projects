"""
Project 678: Agricultural field robot planner
Category: Robotics and Control Projects
Difficulty: Advanced
Domain: robotics and control

Description:
  This project builds a production-minded agricultural field robot planner workflow inside the broader area of robotics and control. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import heapq

PROJECT_NAME = "Agricultural field robot planner"
grid = [
    "S....",
    ".##..",
    "...#.",
    ".#..G",
]

start = goal = None
for r, row in enumerate(grid):
    for c, cell in enumerate(row):
        if cell == "S":
            start = (r, c)
        if cell == "G":
            goal = (r, c)

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def neighbors(node):
    r, c = node
    for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and grid[nr][nc] != "#":
            yield nr, nc

frontier = [(0, start)]
came_from = {start: None}
cost_so_far = {start: 0}

while frontier:
    _, current = heapq.heappop(frontier)
    if current == goal:
        break
    for nxt in neighbors(current):
        new_cost = cost_so_far[current] + 1
        if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
            cost_so_far[nxt] = new_cost
            priority = new_cost + heuristic(nxt, goal)
            heapq.heappush(frontier, (priority, nxt))
            came_from[nxt] = current

path = []
current = goal
while current is not None:
    path.append(current)
    current = came_from[current]
path.reverse()

print(PROJECT_NAME)
print("Planned path:", path)
print("Path length:", len(path) - 1)
