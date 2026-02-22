import json
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import os


WINDOW = 20           
OUTPUT_DIR = "plots"   
os.makedirs(OUTPUT_DIR, exist_ok=True)


with open("training_logs.json", "r") as f:
    data = json.load(f)

# Group data by snapshot
snapshot_curves = defaultdict(lambda: ([], []))

for entry in data["against_all_snapshots"]:
    curr_ep = entry["current_episode"]
    
    for res in entry["results"]:
        snap_ep = res["snapshot_episode"]
        win_rate = res["win_rate"]
        
        snapshot_curves[snap_ep][0].append(curr_ep)
        snapshot_curves[snap_ep][1].append(win_rate)

# ---------- Plot each snapshot separately ----------
for snap_ep, (x, y) in snapshot_curves.items():

    xy = sorted(zip(x, y))
    x_sorted = [p[0] for p in xy]
    y_sorted = [p[1] for p in xy]
    
    # smooth
    y_smooth = pd.Series(y_sorted).rolling(WINDOW, min_periods=1).mean()
    
    # plot
    plt.figure(figsize=(10,5))
    plt.plot(x_sorted, y_smooth, linewidth=2)
    plt.xlabel("Current Episode")
    plt.ylabel("Win Rate")
    plt.title(f"Win Rate vs Snapshot {snap_ep}")
    plt.grid(True)
    
    # save
    save_path = os.path.join(OUTPUT_DIR, f"vs_snapshot_{snap_ep}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

print("✅ All plots saved in:", OUTPUT_DIR)