import tikzplotlib
import matplotlib.pyplot as plt
from rl.rewards import r_pos_e, r_psi_e
import numpy as np
from casadi import Opti

from mpl_toolkits.mplot3d import axes3d
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'text.latex.preamble': [r'\usepackage{lmodern}']
})


def plot3d():
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


def plot_solution(solution: Opti.solve, x: Opti.variable, u: Opti.variable, slack: Opti.variable = None):
    """
    Plot optimizer solution
    """
    dt = 0.05

    x_data = np.round(solution.value(x), 5)[:, :-2]
    u_data = np.round(solution.value(u), 5)
    t_data = np.linspace(0, len(u_data), num=len(u_data[0]))

    # Position plot
    fig0, ax0 = plt.subplots(figsize=(7, 7))
    ax0.plot(x_data[1, :], x_data[0, :])
    ax0.set(xlabel="East", ylabel="North")

    # Position/time plot x
    fig1, ax1 = plt.subplots(figsize=(7, 7))

    ax1.plot(x_data[0, :])
    ax1.set(xlabel="t", ylabel="North")

    # Position/time plot y
    fig2, ax2 = plt.subplots(figsize=(7, 7))

    ax2.plot(x_data[1, :])
    ax2.set(xlabel="t", ylabel="East")

    # Heading plot psi
    fig3, ax3 = plt.subplots(figsize=(7, 7))
    ax3.plot(x_data[2, :])
    ax3.set(xlabel="t", ylabel="$\psi$ heading")

    # U
    fig4, ax4 = plt.subplots(figsize=(7, 7))
    ax4.step(t_data, u_data[0, :],  where="post")
    ax4.step(t_data, u_data[1, :], where="post")
    ax4.set(xlabel="t", ylabel="u control")

    # Print rates
    if len(solution.value(x[:, 0])) > 3:
        fig5, ax5 = plt.subplots(figsize=(7, 7))
        ax5.plot(solution.value(x[3, :]))
        ax5.set(xlabel="t", ylabel="$u$ surge")

        fig5, ax5 = plt.subplots(figsize=(7, 7))
        ax5.plot(x_data[4, :])
        ax5.set(xlabel="t", ylabel="$v$ sway")

        fig5, ax5 = plt.subplots(figsize=(7, 7))
        ax5.plot(x_data[5, :])
        ax5.set(xlabel="t", ylabel="$r$ heading rate")

    if slack is not None:
        s_data = np.round(solution.value(slack), 5)
        pass

    plt.show()
