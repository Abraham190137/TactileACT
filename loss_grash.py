import os
import numpy as np
import matplotlib.pyplot as plt

save_dir = "/home/chenshuai/Project/output/xiaomiloop3"
train_p = os.path.join(save_dir, "graphs", "training_losses.npy")
val_p = os.path.join(save_dir, "graphs", "val_losses.npy")

train = np.load(train_p)
val = np.load(val_p)
print("loaded:", train_p, train.shape)
print("loaded:", val_p, val.shape)

plt.figure(figsize=(10,4))
for i in range(train.shape[1]):
    plt.plot(train[:, i], label=f"cam{i}_train")
    plt.plot(val[:, i], "--", label=f"cam{i}_val")
plt.yscale("log")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("CLIP pretraining (per-camera loss)")
plt.legend(ncol=2)
plt.tight_layout()
plt.show()