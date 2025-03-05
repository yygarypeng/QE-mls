import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import mplhep as hep
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.metrics import r2_score

hep.style.use("ATLAS")


class Plotter:
    def __init__(self):
        pass

    ########################
    # 1D plot
    ########################
    def hist_1d(
        self,
        true,
        pred,
        ranges=[0.0, 1.0],
        xlabel=r"$p_{z}^{\nu\nu}$ [GeV]",
        title="",
        ylabel="Counts",
        sub_ylabel="Pred/True",
        bins=50,
        xpad=10,
        weights=None,
        save_name=None,
        dpi=300,
    ):
        """
        Plot a 1D histogram comparing true and predicted values.

        Parameters:
        :param true: array-like, true values
        :param pred: array-like, predicted values
        :param ranges: list, range of values for the histogram
        :param xlabel: str, label for the x-axis
        :param title: str, title of the plot
        :param ylabel: str, label for the y-axis
        :param sub_ylabel: str, label for the subplot y-axis
        :param bins: int, number of bins for the histogram
        :param xpad: int, padding for the x-axis ticks
        :param weights: array-like, weights for the histogram
        :param save_name: str, name to save the plot
        :param dpi: int, resolution of the saved plot
        """
        fig, (ax1, ax2) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(8, 8),
            gridspec_kw={"height_ratios": [6, 2], "hspace": 0.1},
            sharex=True,
        )

        truth_bar, truth_bin = np.histogram(
            true, bins=bins, range=ranges, weights=weights
        )
        pred_bar, _ = np.histogram(pred, bins=bins, range=ranges, weights=weights)

        hep.histplot(truth_bar, truth_bin, label="True", ax=ax1, lw=2, color="b")
        hep.histplot(pred_bar, truth_bin, label="Pred", ax=ax1, lw=2, color="r")

        label_size = 24
        tick_size = 22
        title_size = 20

        ax2.xaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
        ax1.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
        ax2.xaxis.offsetText.set_fontsize(tick_size)
        ax1.yaxis.offsetText.set_fontsize(tick_size)

        ax1.set_xlim(ranges)
        ax1.legend(fontsize=tick_size)
        ax1.set_ylabel(ylabel, fontsize=label_size, labelpad=5)
        ax1.set_title(title, fontsize=title_size, loc="right")

        ratio = np.divide(pred_bar + 1, truth_bar + 1, where=(truth_bar != 0))
        ax2.vlines(truth_bin[1:], 1, ratio, color="k", lw=1)

        for i, val in enumerate(ratio):
            if val >= 2:
                ax2.annotate(
                    "",
                    xy=(truth_bin[i + 1], 2),
                    xytext=(truth_bin[i + 1], 2),
                    arrowprops=dict(facecolor="k", shrink=0.02, width=1, headwidth=3),
                )
            elif val <= 0:
                ax2.annotate(
                    "",
                    xy=(truth_bin[i + 1], 0),
                    xytext=(truth_bin[i + 1], 0),
                    arrowprops=dict(facecolor="k", shrink=0.02, width=1, headwidth=3),
                )
            else:
                ax2.scatter(truth_bin[i + 1], val, color="k", lw=1, s=10)

        ax2.set_ylim([0, 2])
        ax2.axhline(1, c="grey", ls="dashed", alpha=0.8)

        ax2.set_xlabel(xlabel, fontsize=label_size, labelpad=xpad)
        ax2.set_ylabel(sub_ylabel, fontsize=label_size, loc="center", labelpad=10)

        ax1.tick_params(axis="y", labelsize=tick_size)
        ax2.tick_params(axis="x", labelsize=tick_size, pad=10)
        ax2.tick_params(axis="y", labelsize=tick_size)

        if save_name is not None:
            plt.savefig(f"{save_name}.png", dpi=dpi, bbox_inches="tight")
            print(f"Saved 1D plot as {save_name}.png")

        plt.show()
        plt.close()

    ########################
    # 2D plot
    ########################
    def hist_2d(
        self,
        true,
        pred,
        ranges=[0.0, 1.0],
        xlabel=r"True $p_{z}^{\nu\nu}$ [GeV]",
        ylabel=r"Predicted $p_{z}^{\nu\nu}$ [GeV]",
        title=r"",
        bins=50,
        xpad=10,
        weights=None,
        save_name=None,
        dpi=300,
    ):
        """
        Plot a 2D histogram comparing true and predicted values.

        Parameters:
        :param true: array-like, true values
        :param pred: array-like, predicted values
        :param ranges: list, range of values for the histogram
        :param xlabel: str, label for the x-axis
        :param ylabel: str, label for the y-axis
        :param title: str, title of the plot
        :param bins: int, number of bins for the histogram
        :param xpad: int, padding for the x-axis ticks
        :param weights: array-like, weights for the histogram
        :param save_name: str, name to save the plot
        :param dpi: int, resolution of the saved plot
        """
        if ranges is None:
            ranges = [
                np.min([np.min(pred), np.min(true)]),
                np.max([np.max(pred), np.max(true)]),
            ]

        label_size = 24
        tick_size = 22
        title_size = 20

        fig, ax = plt.subplots(figsize=(8, 8))  # Make the figure square
        h = ax.hist2d(
            pred,
            true,
            bins=bins,
            range=[ranges, ranges],
            cmap="viridis",
            cmin=1,
            norm=LogNorm(),
            weights=weights,
        )
        cbar = fig.colorbar(h[3], ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Frequency", fontsize=tick_size)
        cbar.ax.tick_params(axis="both", which="both", labelsize=tick_size)
        ax.set_title(title, fontsize=title_size, loc="right")
        ax.set_xlabel(xlabel, fontsize=label_size, labelpad=xpad)
        ax.set_ylabel(ylabel, fontsize=label_size, labelpad=5)
        ax.plot(ranges, ranges, color="grey", linestyle="--", alpha=0.8)  # add y=x line
        ax.set_aspect("equal", adjustable="box")
        ax.xaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
        ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
        ax.xaxis.offsetText.set_fontsize(tick_size)
        ax.yaxis.offsetText.set_fontsize(tick_size)
        ax.tick_params(axis="both", labelsize=tick_size, pad=10)
        ax.set_xlim(ranges)
        ax.set_ylim(ranges)

        if save_name is not None:
            plt.savefig(f"{save_name}.png", dpi=dpi, bbox_inches="tight")
            print(f"Saved 2D plot as {save_name}.png")

        plt.show()
        plt.close()

    ########################
    # 1D Grid (2×3)
    ########################
    def hist_1d_grid(
        self,
        true_list,
        pred_list,
        ranges=(0.0, 1.0),
        xlabel="[unit]",
        title="",
        ylabel="Counts",
        sub_ylabel="Pred/True",
        bins=50,
        xpad=8,
        weights=None,
        save_name=None,
        dpi=300,
    ):
        """
        Plot a 2*3 grid of 1D histograms comparing true and predicted values.

        :param true_list: list of array-like, true values
        :param pred_list: list of array-like, predicted values
        :param ranges: list, range of values for the histogram
        :param xlabel: str or list, label for the x-axis
        :param title: str or list, title of the plot
        :param ylabel: str or list, label for the y-axis
        :param sub_ylabel: str or list, label for the subplot y-axis
        :param bins: int, number of bins for the histogram
        :param xpad: int, padding for the x-axis ticks
        :param weights: array-like, weights for the histogram
        :param save_name: str, name to save the plot
        :param dpi: int, resolution of the saved plot
        """

        # Expand single-value params into lists
        if not isinstance(ranges, list):
            ranges = [ranges] * 6
        if not isinstance(xlabel, list):
            xlabel = [xlabel] * 6
        if not isinstance(title, list):
            title = [title] * 6
        if not isinstance(ylabel, list):
            ylabel = [ylabel] * 6
        if not isinstance(sub_ylabel, list):
            sub_ylabel = [sub_ylabel] * 6

        label_size = 18
        tick_size = 14
        title_size = 18

        fig = plt.figure(figsize=(20, 14))
        outer = gridspec.GridSpec(2, 3, wspace=0.1, hspace=0.08)

        for i in range(6):
            # Determine row & col in the 2×3 grid
            row = i // 3  # 0 for top row, 1 for bottom row
            col = i % 3  # 0,1,2 from left to right

            # Nested GridSpec for main (ax1) & ratio (ax2)
            inner = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=outer[i], height_ratios=[6, 2], hspace=0.08
            )
            ax1 = fig.add_subplot(inner[0])
            ax2 = fig.add_subplot(inner[1], sharex=ax1)

            # Remove the top subplot's x tick labels (ratio subplot handles x)
            ax1.tick_params(labelbottom=False)

            # Hist data
            tr_bar, tr_bin = np.histogram(
                true_list[i], bins=bins, range=ranges[i], weights=weights
            )
            pr_bar, _ = np.histogram(
                pred_list[i], bins=bins, range=ranges[i], weights=weights
            )

            # Plot main hist
            hep.histplot(tr_bar, tr_bin, ax=ax1, lw=2, color="b", label="True")
            hep.histplot(pr_bar, tr_bin, ax=ax1, lw=2, color="r", label="Pred")
            ax1.set_xlim(ranges[i])

            # Example: RMSE & SEM in the title
            diff = np.array(pred_list[i]) - np.array(true_list[i])
            rmse = np.sqrt(np.mean(diff**2))
            sem = stats.sem(diff)
            ax1.set_title(
                f"{title[i]} (RMSE={rmse:.2f} ± {sem:.2f})",
                fontsize=title_size,
                loc="right",
            )
            ax1.legend(fontsize=tick_size)

            # Ratio logic with arrow snippet
            ratio = np.divide(pr_bar + 1, tr_bar + 1, where=(tr_bar != 0))
            ax2.vlines(tr_bin[1:], 1, ratio, color="k", lw=1)
            for j, val in enumerate(ratio):
                if val >= 2:
                    ax2.annotate(
                        "",
                        xy=(tr_bin[j + 1], 2),
                        xytext=(tr_bin[j + 1], 2),
                        arrowprops=dict(
                            facecolor="k", shrink=0.02, width=1, headwidth=3
                        ),
                    )
                elif val <= 0:
                    ax2.annotate(
                        "",
                        xy=(tr_bin[j + 1], 0),
                        xytext=(tr_bin[j + 1], 0),
                        arrowprops=dict(
                            facecolor="k", shrink=0.02, width=1, headwidth=3
                        ),
                    )
                else:
                    ax2.scatter(tr_bin[j + 1], val, color="k", lw=1, s=10)

            ax2.set_ylim([0, 2])
            ax2.axhline(1, c="grey", ls="dashed", alpha=0.8)

            # ========== Keep y-labels only for left column =============
            if col == 0:
                # Keep the main hist's y-label
                ax1.set_ylabel(ylabel[i], fontsize=label_size)
                # Keep the ratio subplot's y-label
                ax2.set_ylabel(
                    sub_ylabel[i], fontsize=label_size, loc="center", labelpad=10
                )
            else:
                # Remove y-axis labels
                ax1.set_ylabel("")
                ax2.set_ylabel("")
                ax1.tick_params(labelleft=False)
                ax2.tick_params(labelleft=False)

            # ========== Keep x-labels only for bottom row ==============
            if row == 1:
                ax2.set_xlabel(xlabel[i], fontsize=label_size, labelpad=xpad)
            else:
                ax2.set_xlabel("")
                ax2.tick_params(labelbottom=False)

            # Ticks
            ax1.tick_params(axis="both", labelsize=tick_size)
            ax2.tick_params(axis="both", labelsize=tick_size, pad=10)

        # Shrink the gap between the 6 plots
        plt.subplots_adjust(
            left=0.08, right=0.96, top=0.92, bottom=0.08, wspace=0.1, hspace=0.05
        )

        if save_name:
            plt.savefig(f"{save_name}.png", dpi=dpi, bbox_inches="tight")
            print(f"Saved 1D grid as {save_name}.png")

        plt.show()
        plt.close()

    ########################
    # 2D Grid (2×3)
    ########################
    def hist_2d_grid(
        self,
        true_list,
        pred_list,
        ranges=(0.0, 1.0),
        xlabel="X",
        ylabel="Y",
        title="",
        bins=50,
        xpad=15,
        weights=None,
        save_name=None,
        dpi=300,
    ):
        """
        Plot a 2*3 grid of 2D histograms comparing true and predicted values.

        :param true_list: list of array-like, true values
        :param pred_list: list of array-like, predicted values
        :param ranges: list, range of values for the histogram
        :param xlabel: str or list, label for the x-axis
        :param ylabel: str or list, label for the y-axis
        :param title: str or list, title of the plot
        :param bins: int, number of bins for the histogram
        :param xpad: int, padding for the x-axis ticks
        :param weights: array-like, weights for the histogram
        :param save_name: str, name to save the plot
        :param dpi: int, resolution of the saved plot
        """
        # Expand single-value params
        if not isinstance(ranges, list):
            ranges = [ranges] * 6
        if not isinstance(xlabel, list):
            xlabel = [xlabel] * 6
        if not isinstance(ylabel, list):
            ylabel = [ylabel] * 6
        if not isinstance(title, list):
            title = [title] * 6

        label_size = 18
        tick_size = 14
        title_size = 18

        fig = plt.figure(figsize=(20, 14))
        # 2 rows x 4 columns (last col for colorbar)
        gs = gridspec.GridSpec(
            2, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.12, hspace=0.01
        )

        axes = []
        for i in range(6):
            row = i // 3
            col = i % 3
            ax = fig.add_subplot(gs[row, col])
            axes.append(ax)

        last_h = None
        for i, ax in enumerate(axes):
            row = i // 3
            col = i % 3

            h = ax.hist2d(
                pred_list[i],
                true_list[i],
                bins=bins,
                range=[ranges[i], ranges[i]],
                cmap="viridis",
                cmin=1,
                norm=LogNorm(),
                weights=weights,
            )
            last_h = h[3]  # for colorbar

            # Diagonal
            ax.plot(ranges[i], ranges[i], "k--", alpha=0.7)

            # R2
            r2_val = r2_score(true_list[i], pred_list[i])
            ax.set_title(
                rf"{title[i]} ($R^2$={r2_val:.2f})", fontsize=title_size, loc="right"
            )

            # Only left col has y-label
            if col == 0:
                ax.set_ylabel(ylabel[i], fontsize=label_size)
            else:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)

            # Only bottom row has x-label
            if row == 1:
                ax.set_xlabel(xlabel[i], fontsize=label_size, labelpad=xpad)
                ax.tick_params(axis="both", labelsize=tick_size, pad=10)
            else:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)

            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(ranges[i])
            ax.set_ylim(ranges[i])
            ax.tick_params(axis="both", labelsize=tick_size)
            ax.xaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
            ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))

        # Colorbar on last column
        cax = fig.add_subplot(gs[:, 3])
        cbar = fig.colorbar(last_h, cax=cax, orientation="vertical")
        cbar.ax.text(
            2.5,
            28.0,
            "Frequency",
            va="top",
            ha="left",
            rotation=-90,
            fontsize=label_size,
        )
        cbar.ax.tick_params(labelsize=tick_size)

        # Shrink gap
        plt.subplots_adjust(
            left=0.08, right=0.96, top=0.92, bottom=0.08, wspace=0.12, hspace=0.01
        )

        if save_name:
            plt.savefig(f"{save_name}.png", dpi=dpi, bbox_inches="tight")
            print(f"Saved 2D grid as {save_name}.png")

        plt.show()
        plt.close()


if __name__ == "__main__":
    # Example usage
    plot = Plotter()

    # test individual plots
    pred = np.random.normal(0, 1, 10000)
    truth = np.random.rand(10000)
    plot.hist_1d(
        pred,
        truth,
        ranges=(0, 10000),
        xpad=25,
        save_name="/root/work/QE-mls/qe/1d_test",
    )
    plot.hist_2d(
        pred,
        truth,
        ranges=(0, 10000),
        xpad=25,
        save_name="/root/work/QE-mls/qe/2d_test",
    )

    # test grid plots
    # 1D data with -1 to 1 range
    true_list_1d = [np.random.uniform(-1, 1, 10000) for _ in range(6)]
    pred_list_1d = [
        np.random.normal(0, 0.5, 10000) for _ in range(6)
    ]  # Normal distribution centered at 0
    ranges_1d = (-1, 1)  # Set range from -1 to 1
    xlabels_1d = r"$\xi$ [None]"
    titles_1d = [
        r"$\xi^{(0)}_{n}$",
        r"$\xi^{(0)}_{r}$",
        r"$\xi^{(0)}_{k}$",
        r"$\xi^{(1)}_{n}$",
        r"$\xi^{(1)}_{r}$",
        r"$\xi^{(1)}_{k}$",
    ]
    plot.hist_1d_grid(
        true_list_1d,
        pred_list_1d,
        ranges=ranges_1d,
        xlabel=xlabels_1d,
        title=titles_1d,
        bins=30,
        save_name="/root/work/QE-mls/qe/1d_grid_test",
    )

    # 2D data with -1 to 1 range
    true_list_2d = [np.random.uniform(-1, 1, 10000) for _ in range(6)]
    pred_list_2d = [np.random.normal(0, 0.5, 10000) for _ in range(6)]
    ranges_2d = (-1, 1)  # Set range from -1 to 1
    xlabels_2d = r"True $\xi$ [None]"
    ylabels_2d = r"Predicted $\xi$ [None]"
    titles_2d = [
        r"$\xi^{(0)}_{n}$",
        r"$\xi^{(0)}_{r}$",
        r"$\xi^{(0)}_{k}$",
        r"$\xi^{(1)}_{n}$",
        r"$\xi^{(1)}_{r}$",
        r"$\xi^{(1)}_{k}$",
    ]
    plot.hist_2d_grid(
        true_list_2d,
        pred_list_2d,
        ranges=ranges_2d,
        xlabel=xlabels_2d,
        ylabel=ylabels_2d,
        title=titles_2d,
        bins=30,
        save_name="/root/work/QE-mls/qe/2d_grid_test",
    )

    print("Done!")
