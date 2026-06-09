import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

df = pd.read_csv(r"data/training.csv")

steps = df["step_index"]
loss = df["loss"]
accuracy = df["mean_token_accuracy"]
margin = df["rewards/margins"]


plt.style.use("dark_background")
fig, axes = plt.subplots(1,3, figsize = (16,4))
fig.suptitle("DPO Training — Mistral-7B · hh-rlhf · 5k samples · 3 epochs", fontsize=11, y=1.02)
axes[0].set_title("Train Loss")
axes[0].set_xlabel("Step")
axes[0].plot(steps, loss, color="#7c6af5", linewidth=1.5)
axes[0].grid(alpha=0.15)

axes[1].set_title("Token Accuracy")
axes[1].set_xlabel("Step")
axes[1].plot(steps, accuracy, color="#4ecdc4", linewidth=1.5)
axes[1].grid(alpha=0.15)

axes[2].set_title("Rewards / Margins")
axes[2].set_xlabel("Step")
axes[2].plot(steps, margin, color="#f7dc6f", linewidth=1.5)
axes[2].grid(alpha=0.15)
plt.savefig(r"assets/curves.png",dpi = 300, bbox_inches = "tight")
plt.show()

