import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import mplhep as hep

hep.style.use("ATLAS")


class Plotter:
    def __init__(self):
        pass

    def plot_loss_history(self, history, save_name=None):
        fig = plt.figure(figsize=(8, 5), dpi=120)
        plt.plot(history.history["loss"], lw=2.5, label="Train", alpha=0.8)
        plt.plot(history.history["val_loss"], lw=2.5, label="Validation", alpha=0.8)
        plt.title("Epoch vs MSE", fontsize=16)
        plt.xlabel("epoch", fontsize=14)
        plt.ylabel("Loss (MSE)", fontsize=14)
        plt.legend(loc="best")
        if save_name is not None:
            plt.savefig(save_name)
        plt.show()
        plt.close()
        plt.clf()

    def plot_hist(
        self,
        data=None,
        label=None,
        title=r"Normalized $p^{miss}_{x}$ of MET",
        range=None,
        unit="GeV",
        save_name=None,
    ):
        if range is None:
            min_val = np.min(data[0])
            max_val = np.max(data[0])
            range_val = [min_val, max_val]
        else:
            range_val = range

        n_bins = 50
        fig, axs = plt.subplots(
            2,
            1,
            figsize=(6, 6),
            dpi=100,
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.1},
        )
        arr0 = axs[0].hist(
            data[0],
            bins=n_bins,
            histtype="step",
            density=True,
            color="red",
            linewidth=2,
            label=label[0],
            range=range_val,
        )
        arr1 = axs[0].hist(
            data[1],
            bins=n_bins,
            histtype="step",
            density=True,
            color="blue",
            linewidth=2,
            label=label[1],
            range=range_val,
        )
        axs[0].legend()
        axs[0].set_title(title, fontsize=20)
        axs[0].tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
        axs[0].set_ylabel("(Normalized) counts", fontsize=14)

        ratio = np.divide(arr1[0], arr0[0], where=(arr0[0] != 0))
        axs[1].set_xlabel(label[0] + f" [{unit}]", fontsize=14)
        axs[1].plot(arr0[1][:-1], ratio, "--", color="black", linewidth=1)
        axs[1].axhline(y=1, color="grey", linestyle="--", alpha=0.5)
        axs[1].set_ylabel("ratio", fontsize=14)
        axs[1].tick_params(axis="x", which="both", pad=10)
        if save_name is not None:
            plt.savefig(save_name)
        plt.show()
        plt.close()
        plt.clf()

    def plot_2d_histogram(
        self, pred, truth, title, save_name=None, bins=150, range=None
    ):
        if range is None:
            range = [np.min([pred, truth]), np.max([pred, truth])]
        fig = plt.figure(figsize=(7, 6), dpi=120)
        plt.hist2d(
            pred,
            truth,
            bins=bins,
            range=[range, range],
            cmap="viridis",
            cmin=1,
            norm=LogNorm(),
        )
        cbar = plt.colorbar()
        cbar.set_label("Frequency", fontsize=12)
        plt.title(title, fontsize=16)
        plt.xlabel("Truth", fontsize=12)
        plt.ylabel("Prediction", fontsize=12)
        plt.plot(range, range, color="grey", linestyle="--", alpha=0.8)  # add y=x line
        plt.xlim(range)
        plt.ylim(range)
        plt.gca().set_aspect("equal", adjustable="box")
        if save_name is not None:
            plt.savefig(save_name)
        plt.show()
        plt.close()
        plt.clf()


if __name__ == "__main__":
    plotter = Plotter()

    # Test plot_loss_history
    class History:
        def __init__(self):
            self.history = {
                "loss": [0.1, 0.2, 0.3, 0.4, 0.5],
                "val_loss": [0.2, 0.3, 0.4, 0.5, 0.6],
            }

    history = History()
    plotter.plot_loss_history(history)

    # Test hist
    data = [np.random.normal(0, 1, 1000), np.random.normal(0, 1, 1000)]
    label = ["Data 1", "Data 2"]
    plotter.plot_hist(data, label)

    # Test plot_2d_histogram
    pred = np.random.normal(0, 1, 1000)
    truth = np.random.normal(0, 1, 1000)
    plotter.plot_2d_histogram(pred, truth, "2D Histogram", "2d_hist.png")
