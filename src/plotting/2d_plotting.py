import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.transforms import Affine2D
import numpy as np

from rl.rewards import r_heading

# Latex settings for plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{lmodern,amsmath,amsfonts}')

fig, ax = plt.subplots(figsize=(7, 7))
ax.set_aspect("equal")

pos = (0, 0)
psi = np.pi/8
file_name = "1d_gaussian"

if True:
    sigma = np.pi/8    # [rad]
    C = 0.5            # Max. along axis reward

    x = np.arange(-np.pi/2, np.pi/2, 0.5)
    y = C*np.exp(-1/(2*sigma**2) * x**2)

ax.set(xlim=(-20, 20), ylim=(-20, 20),
       xlabel='E', ylabel='N')

plt.savefig(f'figures/{file_name}.pdf', bbox_inches='tight', dpi=400)
plt.show()
