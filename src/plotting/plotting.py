import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.transforms import Affine2D
import matplotlib.ticker as mtick
import utils
from rl.rewards import r_pos_e
import numpy as np
from casadi import Opti

# TODO: Add over all error plots similar to that in "Combining sysid with RL-based MPC"

# Latex settings for plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{lmodern,amsmath,amsfonts}')


class AnyObject:
    pass


class AnotherObject:
    pass


class AThirdObject:
    pass


class OtterHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        sequence = [[-0.4, 1], [-0.3, 0.8], [-0.3, 0.6], [0.3, 0.6], [0.3, 0.8],
                    [0.4, 1], [0.5, 0.8], [0.5, -0.8],
                    [0.4, -1], [0.3, -0.8], [0.3, -0.6], [-0.3, -0.6], [-0.3, -0.8],
                    [-0.4, -1], [-0.5, -0.8], [-0.5, 0.8]]
        scaled_sequence = []
        for point in sequence:
            x = (point[0] + 0.3)*10
            y = (point[1] + 1)*10
            new_point = [y, x]

            scaled_sequence.append(new_point)

        boat = patches.Polygon(
            scaled_sequence, closed=True, edgecolor='#90552a', facecolor='#f4ac67', linewidth=0.5, alpha=1, transform=handlebox.get_transform())
        handlebox.add_artist(boat)
        return boat


class TargetHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        sequence = [[-0.4, 1], [-0.3, 0.8], [-0.3, 0.6], [0.3, 0.6], [0.3, 0.8],
                    [0.4, 1], [0.5, 0.8], [0.5, -0.8],
                    [0.4, -1], [0.3, -0.8], [0.3, -0.6], [-0.3, -0.6], [-0.3, -0.8],
                    [-0.4, -1], [-0.5, -0.8], [-0.5, 0.8]]
        scaled_sequence = []
        for point in sequence:
            x = (point[0] + 0.3)*10
            y = (point[1] + 1)*10
            new_point = [y, x]

            scaled_sequence.append(new_point)

        boat = patches.Polygon(
            scaled_sequence, closed=True, edgecolor='#90552a', facecolor='#ff0028', linewidth=0.5, alpha=0.6, transform=handlebox.get_transform(), linestyle="--")
        handlebox.add_artist(boat)
        return boat


class DoubleArrowHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        scale = 1.8
        arrow_length = 10
        head_length = 2
        head_width = 3
        between_length = 1.5
        double_arrow_sequence = [[0, arrow_length/2],
                                 [head_width/2, arrow_length/2-head_length],
                                 [0, arrow_length/2],
                                 [-head_width/2, arrow_length/2-head_length],
                                 [0, arrow_length/2],
                                 [0, arrow_length/2-between_length],
                                 [head_width/2, arrow_length/2 -
                                  head_length-between_length],
                                 [0, arrow_length/2-between_length],
                                 [-head_width/2, arrow_length/2 -
                                  head_length-between_length],
                                 [0, arrow_length/2-between_length],
                                 [0, -arrow_length/2]]

        scaled_sequence = []
        for point in double_arrow_sequence:
            x = (point[0] + 1.5) * scale
            y = (point[1] + 6) * scale
            new_point = [y, x]

            scaled_sequence.append(new_point)
        # rotation = Affine2D().rotate(-np.pi/2)
        # transform = rotation + handlebox.get_transform()

        arrow = patches.Polygon(
            scaled_sequence, closed=False, edgecolor='#000000', facecolor='#000000', linewidth=0.7, alpha=1, transform=handlebox.get_transform())
        handlebox.add_artist(arrow)
        return arrow


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
    plt.savefig(f'figures/{file_name}.pdf', bbox_inches='tight', dpi=400)

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


def subplot(dt: float, x_pred, u_pred, x_act, u_act, show=False, save_file_name=None):
    # TODO: Add target pose
    # Ensure arrays
    x_pred = np.asarray(x_pred)
    u_pred = np.asarray(u_pred)
    x_act = np.asarray(x_act).T
    u_act = np.asarray(u_act).T

    t_data = np.arange(start=0, stop=(
        x_act.shape[1])*dt, step=dt)
    N = x_pred[0].shape[1]-1

    if t_data.shape[0] > x_act.shape[1]:
        t_data = t_data[:-1]

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

        # Burnt orange #f4ac67
        # Light blue #97d2d4

    axs[3].step(t_data, u_act[0, :], color="#2e7578", where="post")
    axs[3].set(ylabel=r"$u_{port}$")

    axs[4].step(t_data, u_act[1, :], color="#2e7578", where="post")
    axs[4].set(ylabel=r"$u_{stb}$")

    if save_file_name is not None:
        print(f"Saving file to figures/{save_file_name}_subplots.pdf")
        plt.savefig(
            f'figures/{save_file_name}_subplots.pdf',
            bbox_inches='tight', dpi=400
        )

    if show:
        plt.show()
    else:
        return fig, axs


def slack_subplot(dt: float, slack, show=False, save_file_name=None):
    # Ensure arrays
    slack = np.round(np.asarray(slack), 4)

    t_data = np.arange(start=0, stop=(slack.shape[0])*dt, step=dt)
    N = slack[0].shape[1]-1

    fig, axs = plt.subplots(6, 1, sharex=True)

    axs[0].plot(t_data, slack[:, 0, 1])
    axs[0].set(ylabel=r"$x$")

    axs[1].plot(t_data, slack[:, 1, 1])
    axs[1].set(ylabel=r"$y$")

    axs[2].plot(t_data, slack[:, 2, 1])
    axs[2].set(ylabel=r"$u_{\text{port}}$")

    axs[3].plot(t_data, slack[:, 3, 1])
    axs[3].set(ylabel=r"$u_{\text{stb}}$")

    axs[4].plot(t_data, slack[:, 4, 1])
    axs[4].set(ylabel=r"$\Delta u_{\text{port}}$")

    axs[5].plot(t_data, slack[:, 5, 1])
    axs[5].set(ylabel=r"$\Delta u_{\text{stb}}$")

    for i, s in enumerate(slack):
        for j in range(s.shape[0]):
            t_start = i
            t_end = N+i
            interval = t_data[t_start:t_end]
            num = (3*i*s.shape[0]+j*3)/s.shape[0]
            plot_color = (((10+num)/255, (255-num/2)/255, (255-num)/255))
            # axs[j].plot(interval, s[j, :len(interval)],
            #             color=plot_color, linestyle="--", linewidth=1)
            axs[j].scatter(interval, s[j, :len(interval)],
                           color=plot_color, marker="X", linewidth=0.1)

    if save_file_name is not None:
        print(f"Saving file to figures/{save_file_name}_slack_subplots.pdf")
        plt.savefig(
            f'figures/{save_file_name}_slack_subplots.pdf',
            bbox_inches='tight', dpi=400
        )

    if show:
        plt.show()
    else:
        return fig, axs


def theta_subplot(dt: float, theta, actual, show=False, save_file_name=None):
    # Ensure arrays
    theta = np.asarray(theta)
    theta = np.round(theta, 4)
    actual = np.asarray(actual)
    mass = True
    damp = True
    thrust = False
    env = False
    cost = True
    error = True

    t_data = np.arange(start=0, stop=(theta.shape[0])*dt, step=dt)

    # print(f"t_data.shape[0]: {t_data.shape[0]}")
    # print(f"theta.shape[1]: {theta.shape[0]}")

    if t_data.shape[0] > theta.shape[0]:
        t_data = t_data[:-1]

    if mass:
        fig1, axs1 = plt.subplots(6, 1, sharex=True, figsize=(8, 7))
        plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
        for i in range(6):
            # print(f"t_data.shape: {t_data.shape}")
            axs1[i].plot(t_data, np.round(theta[:, i], 5), color="#2e7578")
            axs1[i].hlines(actual[i], t_data[0], t_data[-1],
                           color="#ff0028", linestyle="--")

        axs1[0].set(ylabel=r"$m$")
        axs1[1].set(ylabel=r"$I_z$")
        axs1[2].set(ylabel=r"$x_g$")
        axs1[3].set(ylabel=r"$X_{\dot{u}}$")
        axs1[4].set(ylabel=r"$Y_{\dot{v}}$")
        axs1[5].set(ylabel=r"$N_{\dot{r}}$")

        if save_file_name is not None:
            print(f"Saving file to figures/{save_file_name}_mass.pdf")
            plt.savefig(
                f'figures/{save_file_name}_mass.pdf',
                bbox_inches='tight', dpi=400
            )

    if damp:
        fig2, axs2 = plt.subplots(4, 1, sharex=True)

        for i in range(4):
            axs2[i].plot(t_data, theta[:, i+6], color="#2e7578")
            axs2[i].hlines(actual[i+6], t_data[0], t_data[-1],
                           color="#ff0028", linestyle="--")

        axs2[0].set(ylabel=r"$X_{u}$")
        axs2[1].set(ylabel=r"$Y_{v}$")
        axs2[2].set(ylabel=r"$N_{r}$")
        axs2[3].set(ylabel=r"$N_{\lvert r \rvert r}$")

        if save_file_name is not None:
            print(f"Saving file to figures/{save_file_name}_damp.pdf")
            plt.savefig(
                f'figures/{save_file_name}_damp.pdf',
                bbox_inches='tight', dpi=400
            )
    # TODO: Fix goal model values for parameter plot, make the lines dashed and find a suitable color
    if thrust:
        fig3, axs3 = plt.subplots(2, 1, sharex=True)

        for i in range(2):
            axs3[i].plot(t_data, theta[:, i+6+4], color="#2e7578")
            axs3[i].hlines(actual[i+6+4], t_data[0],
                           t_data[-1], color="#ff0028", linestyle="--")

        axs3[0].set(ylabel=r"$K_{p}$")
        axs3[1].set(ylabel=r"$K_{s}$")

        if save_file_name is not None:
            print(f"Saving file to figures/{save_file_name}_thrust.pdf")
            plt.savefig(
                f'figures/{save_file_name}_thrust.pdf',
                bbox_inches='tight', dpi=400
            )

    if env:
        fig4, axs4 = plt.subplots(3, 1, sharex=True)

        for i in range(3):
            axs4[i].plot(t_data, theta[:, i+6+4+2], color="#2e7578")
            # axs2[i].hlines(actual[i+6+4+2], t_data[0],
            #                t_data[-1], color="#97d2d4")

        axs4[0].set(ylabel=r"$w_1$")
        axs4[1].set(ylabel=r"$w_2$")
        axs4[2].set(ylabel=r"$w_3$")

        if save_file_name is not None:
            print(f"Saving file to figures/{save_file_name}_env.pdf")
            plt.savefig(
                f'figures/{save_file_name}_env.pdf',
                bbox_inches='tight', dpi=400
            )

    if cost:
        fig5, axs5 = plt.subplots(4, 1, sharex=True)

        for i in range(4):
            axs5[i].plot(t_data, theta[:, i+6+4+2+3], color="#2e7578")

        axs5[0].set(ylabel=r"$\lambda_{\theta}$")
        axs5[1].set(ylabel=r"$V_1$")
        axs5[2].set(ylabel=r"$V_2$")
        axs5[3].set(ylabel=r"$V_3$")

        if save_file_name is not None:
            print(f"Saving file to figures/{save_file_name}_cost.pdf")
            plt.savefig(
                f'figures/{save_file_name}_cost.pdf',
                bbox_inches='tight', dpi=400
            )

    if error:
        fig6, axs6 = plt.subplots(sharex=True)
        model_error = []
        for param in theta:
            # print(f"shape: {(actual).shape}")
            # print(f"shape: {(param).shape}")
            # print(f"shape: {(param[:15] - actual).shape}")
            error = np.linalg.norm(param[:15] - actual, 2)/15
            model_error.append(error)
            # print(f"error: {error}")

        axs6.plot(t_data, model_error, color="#2e7578")
        axs6.set(
            ylabel=r"Model error $\lvert \lvert \theta - \theta_d \rvert \rvert$")

        if save_file_name is not None:
            print(f"Saving file to figures/{save_file_name}_model_error.pdf")
            plt.savefig(
                f'figures/{save_file_name}_cost.pdf',
                bbox_inches='tight', dpi=400
            )

    if show:
        plt.show()
    # else:
    #     return fig, axs


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


def double_arrow(pos, psi, scale, ax):
    arrow_length = 13*scale
    head_length = 2*scale
    head_width = 3*scale
    between_length = 1.5*scale
    double_arrow_sequence = [[0, arrow_length/2],
                             [head_width/2, arrow_length/2-head_length],
                             [0, arrow_length/2],
                             [-head_width/2, arrow_length/2-head_length],
                             [0, arrow_length/2],
                             [0, arrow_length/2-between_length],
                             [head_width/2, arrow_length/2 -
                                 head_length-between_length],
                             [0, arrow_length/2-between_length],
                             [-head_width/2, arrow_length/2 -
                                 head_length-between_length],
                             [0, arrow_length/2-between_length],
                             [0, -arrow_length/2]]
    rotation = Affine2D().rotate(-psi)
    translation = Affine2D().translate(pos[0], pos[1])
    arrow = patches.Polygon(
        double_arrow_sequence, closed=False, edgecolor='#000000', facecolor='#000000', linewidth=1)
    transform = rotation + translation + ax.transData
    arrow.set_transform(transform)
    return arrow


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


def safety_bounds(pos, psi, ax):
    buffer = 0.2
    # buffer = 0.0
    sequence = [[buffer+0.5, buffer+1], [buffer+0.5, -buffer-1],
                [-buffer-0.5, -buffer-1], [-buffer-0.5, buffer+1]]
    rotation = Affine2D().rotate(-psi)
    translation = Affine2D().translate(pos[0], pos[1])
    bound = patches.Polygon(
        sequence, closed=True, edgecolor=(62/255, 98/255, 138/255, 1), facecolor=(151/255, 210/255, 212/255, 0), linewidth=0.5, linestyle="--"
    )
    transform = rotation + translation + ax.transData
    bound.set_transform(transform)
    return bound


def target_pose(pos, psi, alpha, ax):
    sequence = [[-0.4, 1], [-0.3, 0.8], [-0.3, 0.6], [0.3, 0.6], [0.3, 0.8],
                [0.4, 1], [0.5, 0.8], [0.5, -0.8],
                [0.4, -1], [0.3, -0.8], [0.3, -0.6], [-0.3, -0.6], [-0.3, -0.8],
                [-0.4, -1], [-0.5, -0.8], [-0.5, 0.8]]
    rotation = Affine2D().rotate(-psi)
    translation = Affine2D().translate(pos[0], pos[1])
    boat = patches.Polygon(
        sequence, closed=True, edgecolor='#90552a', facecolor='#ff0028', linewidth=0.5, alpha=alpha, linestyle="--")
    transform = rotation + translation + ax.transData
    boat.set_transform(transform)
    return boat


def plot_vessel_path(path, show=False, save_file_name=None):
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

    # north, east, psi = path[-1, :]

    for north, east, psi in path:
        pos = (east, north)
        ax.add_patch(otter(pos, psi, alpha=0.3, ax=ax))

    ax.add_patch(safety_bounds(pos, psi, ax=ax))

    ax.legend([q, rest, b, otter((0, 0), np.pi/2, 1, ax), p], ["Permitted area",
                                                               "Restricted area", r'$\mathbb{S}_h$', "ASV", "Path"], loc="upper left")

    ax.set(xlim=(-20, 20), ylim=(-15, 15),
           xlabel='E', ylabel='N')

    if False:
        ax.text(13, 5, r'$\mathbb{S}$', fontsize=12)

    if save_file_name is not None:
        print(f"Saving file to figures/{save_file_name}_vessel_path.pdf")
        plt.savefig(
            f'figures/{save_file_name}_vessel_path.pdf',
            bbox_inches='tight', dpi=400
        )

    if show:
        plt.show()
    else:
        return fig, ax


def show():
    plt.show()


def brattorkaia(path=None, V_c=0, beta_c=0, show=False, save_file_name=None):
    """
    Map plot of the water within Brattørkaia, Trondheim, Norway

    Map dimensions: (218.98098581121982, 171.260755066673)


    """
    fig, ax = plt.subplots(figsize=(7, 7))

    image_file = "plotting/assets/brattora.png"
    image = plt.imread(image_file)
    dimensions = (218.98098581121982, 171.260755066673)
    extent = (
        -dimensions[0]/4, dimensions[0]/4,
        -dimensions[1]/4, dimensions[1]/4
    )
    ax.imshow(image, extent=extent)

    harbour_sequence = [[-42.5, 15],
                        [-12.5, 40],
                        [30, -7.5],
                        [25.5, -16.5],
                        [15, -26],
                        [-2, -30]]
    harbour_bounds = patches.Polygon(
        harbour_sequence, closed=True, edgecolor="r", facecolor="none", linewidth=1, linestyle="--"
    )

    ax.add_patch(harbour_bounds)
    # if view == "inital":
    ax.add_patch(otter((-20.00666667, 23.240456),
                 utils.D2R(137.37324840062468), 1, ax=ax))
    ax.add_patch(target_pose((19.44486, -20.36019),
                 utils.D2R(137.37324840062468), 0.6, ax=ax))
    # TODO: Make target infill a nice red colour

    if abs(V_c) > 0:
        ax.add_patch(double_arrow((-40, -20), beta_c, 0.7, ax))
        ax.legend([harbour_bounds, AnyObject(), AnotherObject(), AThirdObject()],
                  [r'$\mathbb{S}_b$', "ASV", "Target pose", "Ocean Current"],
                  handler_map={AnyObject: OtterHandler(),
                               AnotherObject: TargetHandler(),
                               AThirdObject: DoubleArrowHandler()},
                  bbox_to_anchor=(0.992, 0.992))
    else:
        ax.legend([harbour_bounds, AnyObject(), AnotherObject()],
                  [r'$\mathbb{S}_b$', "ASV", "Target pose"],
                  handler_map={AnyObject: OtterHandler(
                  ), AnotherObject: TargetHandler()},
                  bbox_to_anchor=(0.992, 0.992))

    if path is not None:
        path = np.asarray(path)
        p, = ax.plot(path[:, 1], path[:, 0], color="#2e7578")

        # north, east, psi = path[-1, :]

        for north, east, psi in path:
            pos = (east, north)
            ax.add_patch(otter(pos, psi, alpha=0.3, ax=ax))

        ax.add_patch(otter(pos, psi, alpha=1, ax=ax))
        ax.add_patch(safety_bounds(pos, psi, ax=ax))

    ax.set(xlabel='E', ylabel='N')

    if save_file_name is not None:
        print(f"Saving file to figures/{save_file_name}_brattorkaia.pdf")
        plt.savefig(
            f'figures/{save_file_name}_brattorkaia.pdf',
            bbox_inches='tight', dpi=400
        )

    if show:
        plt.show()
    else:
        return fig, ax


def ravnkloa(path=None, V_c=0, beta_c=0, show=False, save_file_name=None):
    """
    Map plot of the channel by Ravnkloa, Trondheim, Norway

    Map dimensions (656.9629829983534, 513.7822651994743)

    """

    fig, ax = plt.subplots(figsize=(7, 7))

    # image_file = "plotting/assets/ravnkloa.png"
    image_file = "plotting/assets/ravnkloa_close_up.png"
    image = plt.imread(image_file)
    # dimensions = (656.9629829983534, 513.7822651994743)
    dimensions = (218.98681992249377, 171.26075506640066)
    extent = (
        -dimensions[0]/4, dimensions[0]/4,
        -dimensions[1]/4, dimensions[1]/4
    )

    ax.imshow(image, extent=extent)

    # harbour_sequence = [[110, 26],
    #                     [0, 1],
    #                     [115, 80]]

    harbour_sequence = [[47, -10],
                        [30, -14],
                        [-45, -20],
                        [40, 37]]
    harbour_bounds = patches.Polygon(
        harbour_sequence, closed=True, edgecolor="r", facecolor="none", linewidth=1, linestyle="--"
    )

    ax.add_patch(double_arrow((-10, 15), utils.D2R(-130), 0.7, ax))
    ax.add_patch(harbour_bounds)

    ax.add_patch(otter((-30, -15),
                 utils.D2R(50), 1, ax=ax))
    ax.add_patch(target_pose((36.5, -11),
                 utils.D2R(165), 0.6, ax=ax))

    if abs(V_c) > 0:
        ax.add_patch(double_arrow((-10, 15), utils.D2R(-130), 0.7, ax))
        ax.legend([harbour_bounds, AnyObject(), AnotherObject(), AThirdObject()],
                  [r'$\mathbb{S}_b$', "ASV", "Target pose", "Ocean Current"],
                  handler_map={AnyObject: OtterHandler(),
                               AnotherObject: TargetHandler(),
                               AThirdObject: DoubleArrowHandler()},
                  bbox_to_anchor=(0.992, 0.992))
    else:
        ax.legend([harbour_bounds, AnyObject(), AnotherObject()],
                  [r'$\mathbb{S}_b$', "ASV", "Target pose"],
                  handler_map={AnyObject: OtterHandler(
                  ), AnotherObject: TargetHandler()})  # ,
        #   bbox_to_anchor=(0.992, 0.992))

    # ax.set(xlim=(-5, 120), ylim=(-25, 100), xlabel='E', ylabel='N')
    ax.set(xlabel='E', ylabel='N')

    if save_file_name is not None:
        print(f"Saving file to figures/{save_file_name}_ravnkloa.pdf")
        plt.savefig(
            f'figures/{save_file_name}_ravnkloa.pdf',
            bbox_inches='tight', dpi=400
        )
    if show:
        plt.show()


def nidelva(path=None, show=False, save_file_name=None):
    """
    Map plot of a narrow part of Nidelva, Trondheim, Norway

    Map dimensions: (175.18429477615376, 222.63898158632085)

    """
    fig, ax = plt.subplots(figsize=(7, 7))

    image_file = "plotting/assets/nidelva_close.png"
    image = plt.imread(image_file)
    # dimensions = (175.18429477615376, 222.63898158632085)
    dimensions = (218.9803684729781, 171.26075506640066)
    extent = (
        -dimensions[0]/4, dimensions[0]/4,
        -dimensions[1]/4, dimensions[1]/4
    )

    ax.imshow(image, extent=extent)

    harbour_sequence = []
    harbour_bounds = patches.Polygon(
        harbour_sequence, closed=True, edgecolor="r", facecolor="none", linewidth=1, linestyle="--"
    )

    # ax.add_patch(double_arrow((-10, 15), utils.D2R(-130), 0.7, ax))
    ax.add_patch(harbour_bounds)

    # ax.set(xlim=(-20, 20), ylim=(-15, 15),
    #        xlabel='E', ylabel='N')
    ax.set(xlabel='E', ylabel='N')

    if save_file_name is not None:
        print(f"Saving file to figures/{save_file_name}_nidelva.pdf")
        plt.savefig(
            f'figures/{save_file_name}_nidelva.pdf',
            bbox_inches='tight', dpi=400
        )


def plot_huber():
    # TODO: Chose colours and make this real pretty
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")

    delta = 1
    file_name = "pseudo-Huber-plot"

    if True:
        x = np.arange(-20, 20, 0.01)
        y = delta**2 * (np.sqrt(1 + (x**2)/(delta**2)) - 1)
        y_q = 0.5*x**2

    ax.plot(x, y)    # pseudo-Huber
    ax.plot(x, y_q)  # quadratic
    ax.set(xlim=(-3.5, 3.5), ylim=(0, 3))

    plt.savefig(f'figures/{file_name}.pdf', bbox_inches='tight', dpi=400)
    plt.show()


def prediction_error(e, show=False, save_file_name=None):
    # TODO: Make function for plotting both total model error and individual ones
    fig, ax = plt.subplots(figsize=(7, 7))

    # ax.set(xlim=(-20, 20), ylim=(-15, 15),
    #        xlabel='E', ylabel='N')
    ax.set(xlabel='E', ylabel='N')

    if save_file_name is not None:
        print(f"Saving file to figures/{save_file_name}_nidelva.pdf")
        plt.savefig(
            f'figures/{save_file_name}_nidelva.pdf',
            bbox_inches='tight', dpi=400
        )


def p():
    # TODO: Make function for plotting mpc cost, i.e. loss
    ...
