import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLOR = "#333333"
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937856"]


def setup() :
    plt.rcParams.update({
        "font.sans-serif": ["SimHei", "Microsoft YaHei"],
        "axes.unicode_minus": False,
        "font.size": 14,
        "axes.titlesize": 17,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "axes.edgecolor": COLOR,
        "axes.labelcolor": COLOR,
        "xtick.color": COLOR,
        "ytick.color": COLOR,
        "text.color": COLOR,
    })


def style_ax(ax, grid = True) :
    ax.tick_params(colors = COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ax.spines.values() :
        s.set_color(COLOR)
    ax.title.set_color(COLOR)
    ax.xaxis.label.set_color(COLOR)
    ax.yaxis.label.set_color(COLOR)
    if grid :
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    else :
        ax.grid(False)