import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import mplhep as hep

hep.style.use("ATLAS")


class Plotter:
    def __init__(self):
        pass

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
        ax2.set_ylabel(sub_ylabel, fontsize=label_size, labelpad=10)

        ax1.tick_params(axis="y", labelsize=tick_size)
        ax2.tick_params(axis="x", labelsize=tick_size, pad=10)
        ax2.tick_params(axis="y", labelsize=tick_size)

        if save_name is not None:
            plt.savefig(f"{save_name}.png", dpi=dpi, bbox_inches="tight")
            print(f"Saved plot as {save_name}.png")

        plt.show()
        plt.close()

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
            print(f"Saved plot as {save_name}.png")

        plt.show()
        plt.close()


if __name__ == "__main__":
    # Example usage
    pred = np.random.normal(0, 1, 10000)
    truth = np.random.rand(10000)
    plot = Plotter()
    plot.hist_1d(pred, truth, ranges=(0, 10000), xpad=25, save_name="/root/work/QE-mls/qe/1d_test")
    plot.hist_2d(pred, truth, ranges=(0, 10000), xpad=25, save_name="/root/work/QE-mls/qe/2d_test")
    print("Done!")
