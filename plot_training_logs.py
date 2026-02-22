import json
import matplotlib.pyplot as plt

# Load data from training_logs.json
with open('training_logs.json', 'r') as file:
    data = json.load(file)

# Extract keys and values for plotting
# Assuming the JSON structure is a dictionary with keys as labels and values as lists
keys = list(data.keys())
values = list(data.values())

# Check if the data has at least two keys for separate plots
if len(keys) < 2:
    raise ValueError("The JSON file must contain at least two keys for separate plots.")

# Plot the first key-value pair
plt.figure(figsize=(10, 6))
plt.plot(values[0], label=keys[0])
plt.title(f"Training Data: {keys[0]}")
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.legend()
plt.grid()
plt.savefig(f"{keys[0]}_plot.png")
plt.close()

# Plot the second key-value pair
# plt.figure(figsize=(10, 6))
# plt.plot(values[1], label=keys[1])
# plt.title(f"Training Data: {keys[1]}")
# plt.xlabel("Epochs")
# plt.ylabel("Value")
# plt.legend()
# plt.grid()
# plt.savefig(f"{keys[1]}_plot.png")
# plt.close()
#
# print("Plots saved as images.")