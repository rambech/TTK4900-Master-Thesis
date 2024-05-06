import tikzplotlib
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.transforms import Affine2D
import utils
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


def plot(dt: float, x_data: np.ndarray, u_data: np.ndarray = None, slack_data: np.ndarray = None):
    t_data = np.arange(start=0, stop=(x_data.shape[1]-1)*dt, step=dt)

    # Position plot
    fig0, ax0 = plt.subplots(figsize=(7, 7))
    ax0.plot(x_data[1, :], x_data[0, :])
    ax0.set(xlabel="East", ylabel="North")

    # Position/time plot x
    fig1, ax1 = plt.subplots(figsize=(7, 7))

    ax1.plot(t_data, x_data[0, :-1])
    ax1.set(xlabel="t", ylabel="North")

    # Position/time plot y
    fig2, ax2 = plt.subplots(figsize=(7, 7))

    ax2.plot(t_data, x_data[1, :-1])
    ax2.set(xlabel="t", ylabel="East")

    # Heading plot psi
    fig3, ax3 = plt.subplots(figsize=(7, 7))
    ax3.plot(t_data, x_data[2, :-1])
    ax3.set(xlabel="t", ylabel="$\psi$ heading")

    # U
    fig4, ax4 = plt.subplots(figsize=(7, 7))
    ax4.step(t_data, u_data[0, :],  where="post")
    ax4.step(t_data, u_data[1, :], where="post")
    ax4.set(xlabel="t", ylabel="u control")

    # Print rates
    if len(x_data[:, 0]) > 3:
        fig5, ax5 = plt.subplots(figsize=(7, 7))
        ax5.plot(t_data, x_data[3, :-1])
        ax5.set(xlabel="t", ylabel="$u$ surge")

        fig5, ax5 = plt.subplots(figsize=(7, 7))
        ax5.plot(t_data, x_data[4, :-1])
        ax5.set(xlabel="t", ylabel="$v$ sway")

        fig5, ax5 = plt.subplots(figsize=(7, 7))
        ax5.plot(t_data, x_data[5, :-1])
        ax5.set(xlabel="t", ylabel="$r$ heading rate")

    if slack_data is not None:
        pass

    plt.show()


def subplot(dt: float, x_pred, u_pred, x_act, u_act, save_file_name=None):
    # Ensure arrays
    x_pred = np.asarray(x_pred)
    u_pred = np.asarray(u_pred)
    x_act = np.asarray(x_act).T
    u_act = np.asarray(u_act).T

    t_data = np.arange(start=0, stop=(x_act.shape[1])*dt, step=dt)
    N = x_pred[0].shape[1]-1

    fig, axs = plt.subplots(5, 1, sharex=True)  # layout='constrained'

    for i, x in enumerate(x_pred):
        for j in range(x.shape[0]-3):
            t_start = i
            t_end = N+i
            interval = t_data[t_start:t_end]
            axs[j].plot(interval, x[j, :len(interval)],
                        color="#97d2d4", linestyle="--", linewidth=1)

    axs[0].plot(t_data, x_act[0, :], color="#2e7578")
    axs[0].set(ylabel="North")

    axs[1].plot(t_data, x_act[1, :], color="#2e7578")
    axs[1].set(ylabel="East")

    axs[2].plot(t_data, x_act[2, :], color="#2e7578")
    axs[2].set(ylabel="Heading")

    for i, u in enumerate(u_pred):
        t_start = i
        t_end = N+i
        interval = t_data[t_start:t_end]
        axs[3].step(interval, u[0, :len(interval)],
                    color="#97d2d4", where="post", linestyle="--", linewidth=1)
        axs[4].step(interval, u[1, :len(interval)],
                    color="#97d2d4", where="post", linestyle="--", linewidth=1)

    axs[3].step(t_data, u_act[0, :], color="#2e7578", where="post")
    axs[3].set(ylabel=r"$u_{port}$")

    axs[4].step(t_data, u_act[1, :], color="#2e7578", where="post")
    axs[4].set(ylabel=r"$u_{stb}$")

    if save_file_name is not None:
        plt.savefig(
            f'figures/{save_file_name}_subplots.pdf',
            bbox_inches='tight'
        )
    plt.show()


def plot_solution(dt: float, solution: Opti.solve, x: Opti.variable, u: Opti.variable, slack: Opti.variable = None):
    """
    Plot optimizer solution
    """

    x_data = np.round(solution.value(x), 5)
    u_data = np.round(solution.value(u), 5)

    if slack is not None:
        slack_data = np.round(solution.value(slack), 5)
    else:
        slack_data = None

    plot(dt, x_data, u_data, slack_data)


def otter(pos, psi, alpha, ax):
    sequence = [[-0.4, 1], [-0.3, 0.8], [-0.3, 0.6], [0.3, 0.6], [0.3, 0.8],
                [0.4, 1], [0.5, 0.8], [0.5, -0.8],
                [0.4, -1], [0.3, -0.8], [0.3, -0.6], [-0.3, -0.6], [-0.3, -0.8],
                [-0.4, -1], [-0.5, -0.8], [-0.5, 0.8]]
    rotation = Affine2D().rotate(-psi)
    translation = Affine2D().translate(pos[0], pos[1])
    boat = patches.Polygon(
        sequence, closed=True, edgecolor='#90552a', facecolor='#f4ac67', linewidth=0.5, alpha=alpha)
    transform = rotation + translation + ax.transData
    boat.set_transform(transform)
    return boat


def plot_vessel_path(path, save_file_name=None):
    """
    Function for plotting vessel path on the simple map

    """
    # Latex settings for plot
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('text.latex', preamble=r'\usepackage{lmodern,amsmath,amsfonts}')

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")

    pos = (0, 0)
    # psi = np.pi/8
    target_pos = (0, 15-0.75-0.5)
    target_psi = np.pi/2

    background = patches.Rectangle(
        (-20, -20), 40, 40, edgecolor='#97d2d4', facecolor='#97d2d4', linewidth=1)
    dock = patches.Rectangle((-20, 20-7.5-0.75), 40, 0.75+5,
                             edgecolor='#808080', facecolor='#e6e6e6', linewidth=1)
    quay = patches.Rectangle((-5, 25/2-0.75), 10, 2,
                             edgecolor='#00509e', facecolor='#3e628a', linewidth=1, linestyle="-", alpha=0.3)
    restricted0 = patches.Rectangle((-15, 25/2-0.75), 10, 2,
                                    edgecolor='#595959', facecolor='#000000', linewidth=1, linestyle="-", alpha=0.3)
    restricted1 = patches.Rectangle((5, 25/2-0.75), 10, 2,
                                    edgecolor='#595959', facecolor='#000000', linewidth=1, linestyle="-", alpha=0.3)
    bounds = patches.Rectangle(
        (-15, -25/2), 30, 25, edgecolor="r", facecolor="none", linewidth=1, linestyle="--")

    ax.add_patch(background)
    ax.add_patch(dock)
    ax.add_patch(restricted0)
    rest = ax.add_patch(restricted1)
    q = ax.add_patch(quay)
    b = ax.add_patch(bounds)
    # asv = ax.add_patch(otter(pos, psi))

    path = np.asarray(path)
    p, = ax.plot(path[:, 1], path[:, 0], color="#2e7578")

    for north, east, psi in path:
        pos = (east, north)
        ax.add_patch(otter(pos, psi, alpha=0.3, ax=ax))

    ax.legend([q, rest, b, otter((0, 0), np.pi/2, 1, ax), p], ["Permitted area",
                                                               "Restricted area", r'$\mathbb{S}_h$', "ASV", "Path"], loc="upper left")

    ax.set(xlim=(-20, 20), ylim=(-15, 15),
           xlabel='E', ylabel='N')

    if False:
        ax.text(13, 5, r'$\mathbb{S}$', fontsize=12)

    if save_file_name is not None:
        plt.savefig(
            f'figures/{save_file_name}_vessel_path.pdf',
            bbox_inches='tight'
        )
    plt.show()
