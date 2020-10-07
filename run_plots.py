""" A module that plots Figure 1 to Figure 3 in the paper.

A module that uses the modules activation_functions and supporting_functions
to plot the curves in the paper.

"""

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from activation_functions import list_acti_func
from activation_functions import ReLU
from supporting_functions import square_length
from supporting_functions import simu_square_length
from supporting_functions import sample_qqplot


def set_plot_parameter():
    plt.rcParams.update({'font.size': 15})
    plt.rc('text', usetex=True)
    plt.rc('font', family='sans-serif')


def set_plot_limit(expo, rho):
    """A function that sets ylim and yticks in plot_figure_3.

    Returns:
        ylim, yticks: corresponding to the exponent of lambda and rho.
    """
    d = {}
    d[(-3, 2)] = (0.3, 1.2, 0.1)
    d[(-3, 0.5)] = (0.6, 2.2, 0.2)
    d[(0, 2)] = (0.65, 1.05, 0.05)
    d[(0, 0.5)] = (0.80, 1.025, 0.025)
    d[(-3, 0.2)] = (0.8, 2.6, 0.2)
    yl, yr, yt = d[(expo, rho)] if (expo, rho) in d.keys() else (0, 1.1, 0.1)
    ylim = (yl, yr)
    yticks = np.arange(yl, yr + yt, yt)
    return ylim, yticks


def set_plot_legend(expo, rho):
    list_a = []
    if (expo, rho) in list_a:
        return "upper right"
    else:
        return "lower right"


def format_plot_exponent(p):
    if abs(p) > 10**(-3):
        return '10^{}'.format('{'+str(p)+'}')
    else:
        return '1'


def set_ax_format(ax):
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.tick_params(direction='in', length=4, width=0.3, which='both')
    ax.grid(linewidth=0.3)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.3)


def plot_figure_3():
    """ A function that plots Figure 3.

    Output:
        save Figure_3_{}_{}.pdf on the disk.

    """
    set_plot_parameter()
    psip_l = [10 ** x for x in np.arange(-2, 2, 0.01)]
    for expo in [-3, 0]:
        lamb = 10 ** expo
        for rho in [5, 2, 0.5, 0.2]:
            figname = "./Figure_3_{}_{}.pdf".format(expo, rho)
            fig = plt.figure()
            ax = plt.Axes(fig, [0, 0, 0.7, 0.7])
            fig.add_axes(ax)
            lines = []
            names = [acti_func.name for acti_func in list_acti_func]
            for af in list_acti_func:
                line, = ax.plot(psip_l,
                                [square_length("+oo", psip, lamb, rho, af=af)
                                    for psip in psip_l],
                                linewidth=1, zorder=2, clip_on=False, ls='-')
                lines.append(line)
            ylim, yticks = set_plot_limit(expo, rho)
            title = '$\lambda =' + format_plot_exponent(expo) \
                    + ',\\rho = ' + str(rho) + '$'
            ax.set(xlabel=r'$\psi_p \approx p / n$',
                   ylabel='', ylim=ylim, yticks=yticks,
                   title=title)
            plt.xscale('log')
            set_ax_format(ax)
            ax.legend(lines, names, loc=set_plot_legend(expo, rho))
            plt.savefig(figname, bbox_inches="tight")
            plt.close()


def plot_figure_2a(n_iter=300):
    """ A function that plots Figure 2a.

    Args:
        n_iter: The number of sample for each boxplot.

    Output:
        Write Figure_2a.pdf on the disk.
    """
    set_plot_parameter()
    psid_list_t = np.arange(0.025, 5, 0.025)
    psid_list_b = np.arange(0, 5, 0.1)
    psip = 1/3
    fig = plt.figure()
    ax = plt.Axes(fig, [0, 0, 0.8, 0.8])
    fig.add_axes(ax)
    line, = ax.plot(psid_list_t,
                    [square_length(psid, psip, af=ReLU)
                        for psid in psid_list_t],
                    linewidth=2, zorder=10, clip_on=False, color="xkcd:azure")
    box3 = ax.boxplot([simu_square_length(psid=psid, psip=psip, n_iter=n_iter)
                      for psid in psid_list_b],
                      positions=psid_list_b,
                      widths=0.02,
                      showmeans=True,
                      meanline=True,
                      meanprops=dict(color="orange"),
                      notch=False,
                      patch_artist=True,
                      boxprops=dict(color="none"),
                      capprops=dict(color="xkcd:azure"),
                      whiskerprops=dict(color="xkcd:azure"),
                      flierprops=dict(markersize=3, marker='+',
                                      markeredgecolor="xkcd:azure"),
                      zorder=10)
    for box in [box3]:
        for key in box.keys():
            for artist in box[key]:
                artist.set_clip_on(False)
    ax.set(xlabel=r'$\psi_d \approx d / n$', xlim=(0, 5),
           xticks=np.arange(0, 5.1, 1),
           ylabel='', ylim=(0, 2.5), yticks=np.arange(0, 3.1, 0.25),
           title='')
    ax.legend([line, box3['means'][0], box3['medians'][0], box3['caps'][0]],
              ['Predicted limit', 'Simulated mean', 'Simulated median',
                  'Simulated boxplot'], loc='upper right')
    set_ax_format(ax)
    fig.savefig("Figure_2a.pdf", bbox_inches="tight")
    plt.close()


def transform_qqplot(list_a):
    """A function of a transformation that is used to perform qqplot.

    Args:
        list_a: A list of real numbers.
    """
    list_a.sort()
    n = len(list_a)
    quantile = [(2 * i + 1) / (2 * n) for i in range(n)]
    lc = scipy.stats.norm.ppf(quantile)
    return (list_a, lc)


def plot_figure_2b(n_iter=300):
    """A function that plots figure 2b.
    """
    set_plot_parameter()
    sample = np.asarray(sample_qqplot(psid=2, psip=1/3, n_iter=n_iter))
    fig = plt.figure()
    ax = plt.Axes(fig, [0, 0, 0.8, 0.8])
    fig.add_axes(ax)
    ax.grid(axis="x", linestyle=':', which="both", linewidth=0.3)
    ax.grid(axis="y", linestyle='-', which="both", linewidth=0.1)
    xylim = (-3.0, 3.001)
    xyticks = np.arange(-3.0, 3.1, 1.0)
    ax.tick_params(direction='in', length=4, width=0.3, which='both')
    ax.set(xlabel='Theoretical normal quantile', xlim=xylim, xticks=xyticks,
           ylabel='Sample quantile', ylim=xylim, yticks=xyticks,
           title='')
    l, c = transform_qqplot(sample)
    ax.scatter(c, l, alpha=0.5, marker='+', color="xkcd:azure")
    l2 = np.arange(-4, 4, 0.01)
    line, = ax.plot(l2, l2, color='red', linewidth=0.3)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.3)
    plt.savefig("Figure_2b.pdf", bbox_inches="tight")
    plt.close()


def plot_figure_1a():
    """ A function that plots Figure 1a.

    Output:
        save Figure_1a.pdf on the disk.

    """
    set_plot_parameter()
    step = 0.01
    psid_l = [10 ** x for x in np.arange(-2+step, 2+step, step, dtype=float)]
    psip_l = [0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10]
    lines = []
    N = len(psip_l)
    temp_cycler = plt.rcParams["axes.prop_cycle"]
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, N)))
    fig = plt.figure()
    ax = plt.Axes(fig, [0, 0, 0.7, 0.7])
    fig.add_axes(ax)
    for psip in psip_l:
        line, = ax.plot(psid_l, [square_length(psid, psip, af=ReLU, lamb=10 ** (-2)) for psid in psid_l])
        lines.append(line)
    ax.set(xlabel=r'$\psi_d \approx d / n$',
           ylabel='',
           title='')
    plt.xscale('log')
    plt.ylim((0, 2))
    plt.yticks([0, 0.5, 1, 1.5, 2])
    plt.legend(lines, ['$\psi_p=' + '{:g}'.format(psip) + '$' for psip in psip_l], loc="upper right")
    set_ax_format(ax)
    fig.savefig("Figure_1a.pdf", bbox_inches="tight")
    plt.rcParams["axes.prop_cycle"] = temp_cycler
    plt.close()


def plot_figure_1b():
    """ A function that plots Figure 1b.

    Output:
        save Figure_1b.pdf on the disk.

    """
    set_plot_parameter()
    step = 0.01
    psid_l = [10 ** x for x in np.arange(-2+step, 2+step, step, dtype=float)]
    lamb_l = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]
    N = len(lamb_l)
    temp_cycler = plt.rcParams["axes.prop_cycle"]
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(1, 0, N)))
    psip = 1
    lines = []
    fig = plt.figure()
    ax = plt.Axes(fig, [0, 0, 0.7, 0.7])
    fig.add_axes(ax)
    for lamb in lamb_l:
        line, = ax.plot(psid_l, [square_length(psid, psip, lamb=lamb, af=ReLU) for psid in psid_l], zorder=- np.log(lamb) / 10)
        lines.append(line)
    ax.set(xlabel=r'$\psi_d \approx d / n$',
           ylabel='',
           title='')
    plt.xscale('log')
    plt.ylim((0, 2))
    plt.yticks([0, 0.5, 1, 1.5, 2])
    ax.tick_params(direction='in', length=4, width=0.3, which='both')
    ax.grid(linewidth=0.3)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.3)
    plt.legend(lines, ['$\lambda=' + '{:g}'.format(lamb) + '$' for lamb in lamb_l], loc="upper right")
    ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    fig.savefig("Figure_1b.pdf", bbox_inches="tight")
    plt.rcParams["axes.prop_cycle"] = temp_cycler
    plt.close()


if __name__ == "__main__":
    plot_figure_1a()
    plot_figure_1b()
    plot_figure_2a(n_iter=300)
    plot_figure_2b(n_iter=300)
    plot_figure_3()
