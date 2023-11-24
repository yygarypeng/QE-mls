import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import mplhep as hep

import unittest

hep.style.use("ATLAS")


class Plotter:
    def __init__(self):
        pass

    def plot_loss_history(self, history, logy=False, save_name=None):
        fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
        ax.plot(history.history["loss"], lw=2.5, label="Train", alpha=0.8)
        ax.plot(history.history["val_loss"], lw=2.5, label="Validation", alpha=0.8)
        ax.set_title("Epoch vs MSE", fontsize=16)
        ax.set_xlabel("epoch", fontsize=14)
        ax.set_ylabel("Loss (MSE)", fontsize=14)
        ax.legend(loc="best")
        ax.tick_params(axis="both", labelsize=12)
        if logy is True:
            ax.set_yscale("log")
        if save_name is not None:
            plt.savefig(save_name)
        plt.show()
        plt.close()

    def plot_hist(
        self,
        data=None,
        label=None,
        title=r"Normalized $p^{miss}_{x}$ of MET",
        range=None,
        unit="GeV",
        save_name=None,
    ):
        # Determine the range of the histogram
        data_min, data_max = np.min(data[0]), np.max(data[0])
        range_val = range if range else [data_min, data_max]

        # Create the subplots
        fig, axs = plt.subplots(
            2,
            1,
            figsize=(6, 6),
            dpi=100,
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.1},
        )

        # Plot the histograms
        arr0 = axs[0].hist(
            data[0],
            bins=50,
            histtype="step",
            density=True,
            color="red",
            linewidth=2,
            label=label[0],
            range=range_val,
        )
        arr1 = axs[0].hist(
            data[1],
            bins=50,
            histtype="step",
            density=True,
            color="blue",
            linewidth=2,
            label=label[1],
            range=range_val,
        )

        # Set the title, labels, and tick parameters for the first subplot
        axs[0].set_title(title, fontsize=20)
        axs[0].set_ylabel("(Normalized) counts", fontsize=14)
        axs[0].tick_params(
            axis="x",
            which="both",
            bottom=False,
            top=False,
            labelbottom=False,
            labelsize=12,
        )
        axs[0].legend()

        # Calculate the ratio of the two histograms
        ratio = np.divide(arr1[0], arr0[0], where=(arr0[0] != 0))

        # Plot the ratio and set the labels and tick parameters for the second subplot
        axs[1].plot(arr0[1][:-1], ratio, "--", color="black", linewidth=1)
        axs[1].axhline(y=1, color="grey", linestyle="--", alpha=0.5)
        axs[1].set_xlabel(label[0] + f" [{unit}]", fontsize=14)
        axs[1].set_ylabel("ratio", fontsize=14)
        axs[1].tick_params(axis="x", which="both", pad=10, labelsize=12)

        # Save the figure if a save name is provided
        if save_name:
            plt.savefig(save_name)

        # Show and close the plot
        plt.show()
        plt.close()

    def plot_2d_histogram(
        self, pred, truth, title, save_name=None, bins=150, range=None
    ):
        if range is None:
            range = [
                np.min([np.min(pred), np.min(truth)]),
                np.max([np.max(pred), np.max(truth)]),
            ]
        fig, ax = plt.subplots(figsize=(7, 6), dpi=120)
        h = ax.hist2d(
            pred,
            truth,
            bins=bins,
            range=[range, range],
            cmap="viridis",
            cmin=1,
            norm=LogNorm(),
        )
        cbar = fig.colorbar(h[3], ax=ax)
        cbar.set_label("Frequency", fontsize=12)
        cbar.ax.tick_params(axis="both", which="both", labelsize=12)
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Truth", fontsize=12)
        ax.set_ylabel("Prediction", fontsize=12)
        ax.plot(range, range, color="grey", linestyle="--", alpha=0.8)  # add y=x line
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(axis="both", labelsize=12)
        ax.set_xlim(range)
        ax.set_ylim(range)
        if save_name is not None:
            plt.savefig(save_name)
        plt.show()
        plt.close()


# Test codes


class History:
    def __init__(self, loss, val_loss):
        self.history = {"loss": loss, "val_loss": val_loss}


class TestPlotter(unittest.TestCase):
    def setUp(self):
        self.plotter = Plotter()
        self.data = [np.random.normal(0, 1, 1000), np.random.normal(0, 1, 1000)]
        self.labels = ["Data 1", "Data 2"]
        self.history = History(np.random.rand(10), np.random.rand(10))

    def test_plot_loss_history(self):
        # Just check if the method runs without errors
        self.plotter.plot_loss_history(self.history, logy=True)

    def test_plot_hist(self):
        # Just check if the method runs without errors
        self.plotter.plot_hist(self.data, self.labels)

    def test_plot_2d_histogram(self):
        # Just check if the method runs without errors
        self.plotter.plot_2d_histogram(self.data[0], self.data[1], "2D Histogram")


if __name__ == "__main__":
    unittest.main()
