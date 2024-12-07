import adp3d
from chroma import Protein
import sys
import os

print("Running with args: ", sys.argv)
structure_file = sys.argv[1]
gt_file = sys.argv[2]
density_file = sys.argv[3]
protein = Protein(gt_file)
_, _, S = protein.to_XCS()
adp = adp3d.ADP3D(density_file, S, structure_file, all_atom = True, em = True)

epochs = 4000

test_prot, loss_m, loss_d, loss_s = adp.model_refinement_optimizer(epochs=epochs, output_dir=sys.argv[4])

test_prot.to_PDB(os.path.join(sys.argv[4], 'final.pdb'))

# save loss plots
import matplotlib.pyplot as plt

epoch_timeline = range(epochs)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
fig.suptitle('Loss per Epoch')

ax1.plot(epoch_timeline, loss_m, color='blue')
ax1.set_ylabel('Incomplete structure loss')
ax1.grid(True, linestyle='--', alpha=0.7)

ax2.plot(epoch_timeline, loss_d, color='red')
ax2.set_ylabel('Density loss')
ax2.grid(True, linestyle='--', alpha=0.7)

ax3.plot(epoch_timeline, loss_s, color='green')
ax3.set_xlabel('Epochs')
ax3.set_ylabel('Sequence loss')
ax3.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(sys.argv[4], 'losses.png'))

# save metrics
import numpy as np

metrics = np.array([loss_m, loss_d, loss_s])
np.savetxt(os.path.join(sys.argv[4], 'metrics.csv'), metrics, delimiter=',', header='loss_m,loss_d,loss_s', comments='')