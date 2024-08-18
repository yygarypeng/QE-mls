import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import mplhep as hep

import unittest

hep.style.use("ATLAS")


class Plotter:
    def __init__(self):
        pass

    def plot_loss_history(self, history, logy=False, logx=False, loss_name="MSE", save_name=None):
        # Create the subplots
        fig, axs = plt.subplots(
            1,  # if plotting accuracy, change to 2
            1,
            figsize=(6, 6),
            dpi=100,
            sharex=True,
            # gridspec_kw={"height_ratios": [1, 1], "hspace": 0.1},
        )
        # axs[0].plot(history.history["loss"], lw=2.5, label="Train", alpha=0.8)
        # axs[0].plot(history.history["val_loss"], lw=2.5, label="Validation", alpha=0.8)
        axs.plot(history.history["loss"], lw=2.5, label="Train", alpha=0.8)
        axs.plot(history.history["val_loss"], lw=2.5, label="Validation", alpha=0.8)
        # axs[1].plot(history.history["accuracy"], lw=2.5, label="Train", alpha=0.8)
        # axs[1].plot(
        #     history.history["val_accuracy"], lw=2.5, label="Validation", alpha=0.8
        # )

        axs.set_title("Learning curves", fontsize=16)
        axs.set_ylabel(f"Loss {loss_name}", fontsize=14)
        axs.set_xlabel("epoch", fontsize=14)
        # axs[0].set_title("Learning curves", fontsize=16)
        # axs[0].set_ylabel("Loss (MSE)", fontsize=14)
        # axs[1].set_xlabel("epoch", fontsize=14)
        # axs[1].set_ylabel("Accuracy", fontsize=14)

        axs.legend(loc="best")
        axs.tick_params(axis="both", which="both", labelsize=10)
        # axs[0].legend(loc="best")
        # axs[0].tick_params(axis="both", which="both", labelsize=10)
        # axs[1].tick_params(axis="both", which="both", labelsize=10)

        if logy is True:
            axs.set_yscale("log")
        if logx is True:
            axs.set_xscale("log")
            # axs[0].set_yscale("log")
            # axs[1].set_yscale("log")

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
        xlabel="GeV",
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
            bottom=True,
            top=True,
            labelbottom=False,
            labelsize=10,
        )
        axs[0].tick_params(
            axis="y",
            which="both",
            labelsize=10,
        )
        axs[0].legend()

        # Calculate the ratio of the two histograms
        ratio = np.divide(arr1[0], arr0[0], where=(arr0[0] != 0))

        # Plot the ratio and set the labels and tick parameters for the second subplot
        axs[1].plot(arr0[1][:-1], ratio, "--", color="black", linewidth=1)
        axs[1].axhline(y=1, color="grey", linestyle="--", alpha=0.5)
        axs[1].set_xlabel(f"{xlabel}", fontsize=14)
        axs[1].set_ylabel("ratio", fontsize=14)
        axs[1].tick_params(axis="x", which="both", pad=10, labelsize=10)
        axs[1].tick_params(axis="y", which="both", labelsize=10)

        # Save the figure if a save name is provided
        if save_name:
            plt.savefig(save_name)

        # Show and close the plot
        plt.show()
        plt.close()

    def hist(
        self,
        data,
        range=[0.3, 0.7],
        title=r"$p_{z}^{\nu\nu}$",
        label=r"$p_{z}^{\nu\nu}$",
        unit="[unit]",
    ):

        fig, ax = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(8, 8),
            gridspec_kw={"height_ratios": [6, 2]},
            sharex=True,
            tight_layout=True,
        )
        ax = ax.flatten()
        truth_bar, truth_bin = np.histogram(data[0], bins=50, range=range)
        pred_bar, pred_bin = np.histogram(data[1], bins=50, range=range)
        hep.histplot(
            truth_bar, truth_bin, label="True " + label, ax=ax[0], lw=2, color="b"
        )
        hep.histplot(
            pred_bar,
            truth_bin,
            label="Pred " + label,
            ax=ax[0],
            lw=2,
            color="r",
        )
        epsilon = 1
        ax[0].set_xlim(range)
        ax[0].legend()
        ax[0].set_ylabel("Counts")
        ax[0].set_title(title)
        ratio = np.divide(pred_bar+epsilon, truth_bar+epsilon, where=(truth_bar != 0))
        ax[1].vlines(truth_bin[1::], 1, ratio, color="k", lw=1)
        ax[1].scatter(truth_bin[1::], ratio, color="k", lw=1, s=10, label="")
        # ax[1].set_yscale('log')
        ax[1].set_ylim([0, 2])
        ax[1].axhline(1, c="grey", ls="dashed")
        if unit == "[unit]":
            ax[1].set_xlabel("Scaled " + label + " " + unit)
        else:
            ax[1].set_xlabel(label + " " + unit)
        ax[1].set_ylabel("Pred/True")
        ax[1].tick_params(axis="x", pad=9)
        plt.show()

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
        cbar.ax.tick_params(axis="both", which="both", labelsize=10)
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Truth", fontsize=12)
        ax.set_ylabel("Prediction", fontsize=12)
        ax.plot(range, range, color="grey", linestyle="--", alpha=0.8)  # add y=x line
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(axis="both", labelsize=10)
        ax.set_xlim(range)
        ax.set_ylim(range)
        if save_name is not None:
            plt.savefig(save_name)
        # hep.atlas.label(loc=1)
        plt.show()
        plt.close()


# Test codes
class TestPlotter(unittest.TestCase):
    def setUp(self):
        self.plotter = Plotter()
        self.history = type("", (), {})()
        self.history.history = {
            "loss": np.random.rand(10),
            "val_loss": np.random.rand(10),
            "accuracy": np.random.rand(10),
            "val_accuracy": np.random.rand(10),
        }
        self.data = [np.random.normal(0, 1, 1000), np.random.normal(0, 1, 1000)]
        self.labels = ["Data 1", "Data 2"]
        self.truth = np.random.rand(1000)
        self.pred = np.random.rand(1000)

    def test_plot_loss_history(self):
        try:
            self.plotter.plot_loss_history(self.history, logy=True)
            result = True
        except:
            result = False
        self.assertEqual(result, True)

    def test_plot_hist(self):
        try:
            self.plotter.plot_hist(self.data, self.labels)
            result = True
        except:
            result = False
        self.assertEqual(result, True)

    def test_hist(self):
        try:
            self.plotter.hist(self.data)
            result = True
        except Exception as e:
            print(f"Caught an exception: {e}")
            result = False
        self.assertEqual(result, True)

    def test_plot_2d_histogram(self):
        try:
            self.plotter.plot_2d_histogram(self.truth, self.pred, "2D Histogram")
            result = True
        except:
            result = False
        self.assertEqual(result, True)


if __name__ == "__main__":
    unittest.main()
