import tikzplotlib
import matplotlib.pyplot as plt
from rl.rewards import r_pos_e, r_psi_e
import numpy as np

from mpl_toolkits.mplot3d import axes3d
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'text.latex.preamble': [r'\usepackage{lmodern}']
})

ax = plt.figure().add_subplot(projection='3d')
# ax1 = plt.figure().add_subplot(projection='3d')
x = y = np.arange(-15, 15, 0.1)
X, Y = np.meshgrid(x, y)

Z = r_pos_e((X-10.75, Y))
zeros = np.zeros((Z.shape))

# Plot the 3D surface 'royalblue'
ax.plot_surface(X, Y, Z, edgecolor="#2e7578", linewidth=0.5, rstride=8, cstride=8,
                alpha=0.3)
# ax.plot_surface(X, Y, zeros, edgecolor='#f4ac67', linewidth=0.2, rstride=8, cstride=8,
#                 alpha=0.1)

# Plot projections of the contours for each dimension.  By choosing offsets
# that match the appropriate axes limits, the projected contours will sit on
# the 'walls' of the graph.
# ax.contour(X, Y, Z, zdir='z', offset=-3, cmap='coolwarm')
# ax.contour(X, Y, Z, zdir='x', offset=-20, cmap='coolwarm')
# ax.contour(X, Y, Z, zdir='y', offset=20, cmap='coolwarm')

ax.set(xlim=(-16, 16), ylim=(-16, 16), zlim=(-1, 1),
       xlabel='N', ylabel='E', zlabel='R')


# Plot the 3D surface
# ax1.plot_surface(X, Y, Z1, edgecolor='seagreen', lw=0.5, rstride=8, cstride=8,
#                  alpha=0.3)

# Plot projections of the contours for each dimension.  By choosing offsets
# that match the appropriate axes limits, the projected contours will sit on
# the 'walls' of the graph.
# ax1.contour(X, Y, Z, zdir='z', offset=-100, cmap='coolwarm')
# ax1.contour(X, Y, Z, zdir='x', offset=-40, cmap='coolwarm')
# ax1.contour(X, Y, Z, zdir='y', offset=40, cmap='coolwarm')

# ax1.set(xlim=(-20, 20), ylim=(-20, 20), zlim=(-3, 3),
#         xlabel='X', ylabel='Y', zlabel='Z')

file_name = input("Input file name: ")
plt.savefig(f'figures/{file_name}.pdf', bbox_inches='tight')

plt.show()
