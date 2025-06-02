import json
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl

mpl.rc('font', weight='bold')  # Set all fonts to bold

# Load the JSON file
file_path = "output/head_detection/kl.json"
with open(file_path, "r") as file:
    data_dict = json.load(file)

# Convert string keys to tuple indices
data_dict = {tuple(map(int, k.strip("[]").split(", "))): v for k, v in data_dict.items()}

# Extract max indices to define the heatmap size
x_max = max(k[0] for k in data_dict.keys()) + 1
y_max = max(k[1] for k in data_dict.keys()) + 1

# Create an empty matrix
heatmap = np.zeros((y_max, x_max))

# Fill the matrix with data
for (x, y), value in data_dict.items():
    heatmap[y, x] = value

# Plot the heatmap with reversed colormap to make bigger values red
fig, ax = plt.subplots(figsize=(6, 3))
cax = ax.imshow(heatmap, cmap="coolwarm", aspect="auto")  # Reverse colormap "coolwarm_r"

# Add grid lines for clarity
ax.set_xticks(np.arange(x_max))
ax.set_yticks(np.arange(y_max))
ax.set_xticklabels(np.arange(x_max), fontsize=6, fontweight='bold')
ax.set_yticklabels(np.arange(y_max), fontsize=6, fontweight='bold')
ax.set_xlabel("Layer ID", fontsize=12, fontweight='bold')
ax.set_ylabel("Head ID", fontsize=12, fontweight='bold')
ax.set_title("KL Divergence", fontsize=15, fontweight='bold')

# Show color bar
plt.colorbar(cax)
plt.savefig("output/figures/kl.pdf", dpi=300, bbox_inches="tight")