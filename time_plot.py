import matplotlib.pyplot as plt
import json
import numpy as np

# Read the timings from the file
timings_file = 'timings.txt'

levels = []
step_1_times = []
step_2_times = []
step_4_times = []
step_5_times = []
step_7_times = []

with open(timings_file, 'r') as file:
    for level, line in enumerate(file, start=1):
        timings = json.loads(line.strip())
        levels.append(level)
        step_1_times.append(timings['Step 1: Schur Decomposition'])
        step_2_times.append(timings['Step 2: Transform M'])
        step_4_times.append(timings['Step 3: Precompute LU'])
        step_5_times.append(timings['Step 5: Solve for X~'])
        step_7_times.append(timings['Step 7: Transform X_tilde back to X'])

# Convert to proportion of total time for each level
total_times = np.array(step_1_times) + np.array(step_2_times) + np.array(step_4_times) + np.array(step_5_times) + np.array(step_7_times)
step_1_times = np.array(step_1_times) / total_times
step_2_times = np.array(step_2_times) / total_times
step_4_times = np.array(step_4_times) / total_times
step_5_times = np.array(step_5_times) / total_times
step_7_times = np.array(step_7_times) / total_times

# Create a stacked area plot
plt.figure(figsize=(12, 8))
plt.stackplot(levels, step_1_times, step_2_times, step_4_times, step_5_times, step_7_times, labels=[
    'Step 1: Schur Decomposition', 'Step 2: Transform Q^', 'Step 3: Precompute LU Decompositions', 'Step 4: Solve for X^', 'Step 5: Transform X^ back to X'
])

plt.xlabel('Level')
plt.ylabel('Proportion of Total Time')
plt.title('Proportion of Time Spent in Each Step of the Sparse-Dense Sylvester Solver')
plt.legend(loc='center')
plt.grid(True)
plt.xticks(levels)
plt.show()