import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
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
        legend=["True", "Pred"],
        bins=50,
        xpad=10,
        weights=None,
        save_name=None,
        dpi=300,
    ):
        """
        Plot a 1D histogram comparing true and predicted values.
        """
        fig, (ax1, ax2) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(8, 8),
            gridspec_kw={"height_ratios": [6, 2], "hspace": 0.1},
            sharex=True,
        )

        truth_bar, truth_bin = np.histogram(true, bins=bins, range=ranges, weights=weights)
        pred_bar, _ = np.histogram(pred, bins=bins, range=ranges, weights=weights)

        hep.histplot(truth_bar, truth_bin, label=f"{legend[0]}", ax=ax1, lw=2, color="b")
        hep.histplot(pred_bar, truth_bin, label=f"{legend[1]}", ax=ax1, lw=2, color="r")

        label_size = 24
        tick_size = 22
        title_size = 20

        ax2.xaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
        ax1.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
        ax2.xaxis.offsetText.set_fontsize(tick_size)
        ax1.yaxis.offsetText.set_fontsize(tick_size)

        ax1.set_xlim(ranges)
        ax1.legend(fontsize=title_size)
        ax1.set_ylabel(ylabel, fontsize=label_size, labelpad=5)
        ax1.set_title(title, fontsize=title_size, loc="right")

        ratio = np.divide(pred_bar + 1, truth_bar + 1, where=(truth_bar != 0))
        ax2.vlines(truth_bin[1:], 1, ratio, color="k", lw=1)

        for i, val in enumerate(ratio):
            if val >= 2:
                ax2.annotate(
                    "",
                    xy=(truth_bin[i + 1], 2),
                    xytext=(truth_bin[i + 1], 1.95),
                    arrowprops=dict(facecolor="k", shrink=0.05, width=1, headwidth=2),
                )
            elif val <= 0:
                ax2.annotate(
                    "",
                    xy=(truth_bin[i + 1], 0),
                    xytext=(truth_bin[i + 1], 0.1),
                    arrowprops=dict(facecolor="k", shrink=0.05, width=1, headwidth=2),
                )
            else:
                ax2.scatter(truth_bin[i + 1], val, color="k", lw=1, s=10)

        ax2.set_ylim([0, 2])
        ax2.axhline(1, c="grey", ls="dashed", alpha=0.8)

        ax2.set_xlabel(xlabel, fontsize=label_size, labelpad=xpad)
        ax2.set_ylabel(f"{legend[1]}/{legend[0]}", fontsize=label_size, loc="center", labelpad=10)

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
        log=True,
        weights=None,
        save_name=None,
        dpi=300,
    ):
        """
        Plot a 2D histogram comparing true and predicted values.
        """
        if ranges is None or not isinstance(ranges, (list, tuple)) or len(ranges) != 2:
            ranges = [np.min([np.min(pred), np.min(true)]), np.max([np.max(pred), np.max(true)])]

        label_size = 24
        tick_size = 22
        title_size = 20

        fig, ax = plt.subplots(figsize=(8, 8))
        if log:
            h = ax.hist2d(
                pred, true, bins=bins, range=[ranges, ranges], cmap="viridis", cmin=1,
                norm=LogNorm(), weights=weights,
            )
        else:
            h = ax.hist2d(
                pred, true, bins=bins, range=[ranges, ranges], cmap="viridis", cmin=1,
                weights=weights,
            )
        cbar = fig.colorbar(h[3], ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Counts" if log else "Counts", fontsize=tick_size)
        cbar.ax.tick_params(axis="both", which="both", labelsize=tick_size)
        ax.set_title(title, fontsize=title_size, loc="right")
        ax.set_xlabel(xlabel, fontsize=label_size, labelpad=xpad)
        ax.set_ylabel(ylabel, fontsize=label_size, labelpad=5)
        ax.plot(ranges, ranges, color="grey", linestyle="--", alpha=0.8)
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
        legend_lst=["Pred", "True"],
        bins=50,
        xpad=8,
        rmse_title=False,
        weights=None,
        save_name=None,
        dpi=500,
    ):
        """
        Plot a 2*3 grid of 1D histograms comparing true and predicted values.
        """
        if not isinstance(ranges, list):
            ranges = [ranges] * 6
        if not isinstance(xlabel, list):
            xlabel = [xlabel] * 6
        if not isinstance(title, list):
            title = [title] * 6
        if not isinstance(ylabel, list):
            ylabel = [ylabel] * 6
        sub_ylabel = f"{legend_lst[0]}/{legend_lst[1]}"
        if not isinstance(sub_ylabel, list):
            sub_ylabel = [sub_ylabel] * 6

        label_size = 20
        tick_size = 16
        title_size = 20

        fig = plt.figure(figsize=(20, 14), dpi=dpi)
        outer = gridspec.GridSpec(2, 3, wspace=0.1, hspace=0.08)

        for i in range(6):
            row = i // 3
            col = i % 3
            
            inner = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=outer[i], height_ratios=[6, 2], hspace=0.08
            )
            ax1 = fig.add_subplot(inner[0])
            ax2 = fig.add_subplot(inner[1], sharex=ax1)
            ax1.tick_params(labelbottom=False)

            # Force y-axis to use scientific notation
            formatter = plt.ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-1, 1))  # Force scientific notation for all values
            ax1.yaxis.set_major_formatter(formatter)

            tr_bar, tr_bin = np.histogram(true_list[i], bins=bins, range=ranges[i], weights=weights)
            pr_bar, _ = np.histogram(pred_list[i], bins=bins, range=ranges[i], weights=weights)

            hep.histplot(tr_bar, tr_bin, ax=ax1, lw=2, color="b", label=legend_lst[1])
            hep.histplot(pr_bar, tr_bin, ax=ax1, lw=2, color="r", label=legend_lst[0])
            ax1.set_xlim(ranges[i])
            
            if rmse_title is True:
                diff = np.array(pred_list[i]) - np.array(true_list[i])
                rmse = np.sqrt(np.mean(diff**2))
                sem = stats.sem(diff)
                ax1.set_title(f"{title[i]} (RMSE={rmse:.2f} ± {sem:.2f})", fontsize=title_size, loc="right")
            else:
                ax1.set_title(title[i], fontsize=title_size, loc="right")
            ax1.legend(fontsize=title_size)

            ratio = np.divide(pr_bar + 1, tr_bar + 1, where=(tr_bar != 0))
            ax2.vlines(tr_bin[1:], 1, ratio, color="k", lw=1)
            for j, val in enumerate(ratio):
                if val > 2:
                    ax2.annotate(
                        "", xy=(tr_bin[j + 1], 2), xytext=(tr_bin[j + 1], 1.95),
                        arrowprops=dict(facecolor="k", shrink=0.05, width=1, headwidth=2),
                    )
                elif val < 0:
                    ax2.annotate(
                        "", xy=(tr_bin[j + 1], 0), xytext=(tr_bin[j + 1], 0.05),
                        arrowprops=dict(facecolor="k", shrink=0.05, width=1, headwidth=2),
                    )
                else:
                    ax2.scatter(tr_bin[j + 1], val, color="k", lw=1, s=10)

            ax2.set_ylim([0, 2])
            ax2.axhline(1, c="grey", ls="dashed", alpha=0.8)

            if col == 0:
                ax1.set_ylabel(ylabel[i], fontsize=label_size)
                ax2.set_ylabel(sub_ylabel[i], fontsize=label_size, loc="center", labelpad=10)
            else:
                ax1.set_ylabel("")
                ax2.set_ylabel("")
                ax1.tick_params(labelleft=True)
                ax2.tick_params(labelleft=False)

            if row == 1:
                ax2.set_xlabel(xlabel[i], fontsize=label_size, labelpad=xpad)
            else:
                ax2.set_xlabel("")
                ax2.tick_params(labelbottom=False)

            ax1.tick_params(axis="both", labelsize=tick_size)
            ax2.tick_params(axis="both", labelsize=tick_size, pad=10)

        plt.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.08, wspace=0.1, hspace=0.05)

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
        log=True,
        xpad=15,
        weights=None,
        save_name=None,
        dpi=500,
    ):
        """
        Plot a 2*3 grid of 2D histograms comparing true and predicted values with a shared color bar.
        """
        if not isinstance(ranges, list):
            ranges = [ranges] * 6
        if not isinstance(xlabel, list):
            xlabel = [xlabel] * 6
        if not isinstance(ylabel, list):
            ylabel = [ylabel] * 6
        if not isinstance(title, list):
            title = [title] * 6

        label_size = 20
        tick_size = 16
        title_size = 20

        fig = plt.figure(figsize=(20, 14), dpi=dpi)
        gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.18, hspace=-0.1)

        axes = []
        for i in range(6):
            row = i // 3
            col = i % 3
            ax = fig.add_subplot(gs[row, col])
            axes.append(ax)

        # Ensure ranges are valid
        for i in range(6):
            if not isinstance(ranges[i], (list, tuple)) or len(ranges[i]) != 2:
                ranges[i] = [min(true_list[i]), max(true_list[i])]

        # Find global max/min counts for consistent color scaling
        max_count = 0
        min_count = float('inf')
        for i in range(min(len(true_list), len(pred_list), 6)):
            hist_vals, _, _ = np.histogram2d(
                pred_list[i], true_list[i], bins=bins, range=[ranges[i], ranges[i]], weights=weights
            )
            max_count = max(max_count, np.max(hist_vals))
            non_zero_min = np.min(hist_vals[hist_vals > 0]) if np.any(hist_vals > 0) else 1
            min_count = min(min_count, non_zero_min)
        
        if min_count == float('inf'):
            min_count = 1  # Fallback for empty or all-zero data

        # Shared normalization
        if log:
            norm = LogNorm(vmin=min_count, vmax=max_count)
        else:
            norm = Normalize(vmin=min_count, vmax=max_count)

        # Create the plots with shared normalization
        for i, ax in enumerate(axes):
            row = i // 3
            col = i % 3
            
            formatter = plt.ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-1, 1))  # Force scientific notation for all values
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)
            
            h = ax.hist2d(
                pred_list[i], true_list[i], bins=bins, range=[ranges[i], ranges[i]],
                cmap="viridis", cmin=1, norm=norm, weights=weights,
            )

            ax.plot(ranges[i], ranges[i], "k--", alpha=0.7)
            r2_val = r2_score(true_list[i], pred_list[i])
            ax.set_title(rf"{title[i]} ($R^2$={r2_val:.2f})", fontsize=title_size, loc="right")

            if col == 0:
                ax.set_ylabel(ylabel[i], fontsize=label_size)
            else:
                ax.set_ylabel("")
                ax.tick_params(labelleft=True)

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

        # Shared color bar
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        cax = fig.add_subplot(gs[:, 3])
        cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
        cbar.set_label("Counts" if log else "Counts", fontsize=label_size, labelpad=10)
        cbar.ax.tick_params(labelsize=tick_size)

        plt.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.08, wspace=0.12, hspace=-0.1)

        if save_name:
            plt.savefig(f"{save_name}.png", dpi=dpi, bbox_inches="tight")
            print(f"Saved 2D grid as {save_name}.png")

        plt.show()
        plt.close()


    ########################
    # 1D Grid (4×2)
    ########################
    def hist_1d_grid_4plots(
        self,
        true_list,
        pred_list,
        ranges=(0.0, 1.0),
        xlabel="[unit]",
        title="",
        ylabel="Counts",
        legend_lst=["Pred", "True"],
        bins=50,
        xpad=8,
        rmse_title=False,
        weights=None,
        save_name=None,
        dpi=500,
    ):
        """
        Plot a 4*2 grid of 1D histograms comparing true and predicted values.
        """
        # Ensure we have sufficient data
        n_plots = min(len(true_list), len(pred_list), 8)
        
        if not isinstance(ranges, list):
            ranges = [ranges] * n_plots
        if not isinstance(xlabel, list):
            xlabel = [xlabel] * n_plots
        if not isinstance(title, list):
            title = [title] * n_plots
        if not isinstance(ylabel, list):
            ylabel = [ylabel] * n_plots
        sub_ylabel = f"{legend_lst[0]}/{legend_lst[1]}"
        if not isinstance(sub_ylabel, list):
            sub_ylabel = [sub_ylabel] * n_plots

        label_size = 20
        tick_size = 16
        title_size = 20

        # Change figure size to be taller than wide for 4×2 layout
        fig = plt.figure(figsize=(20, 30), dpi=dpi)
        outer = gridspec.GridSpec(4, 2, wspace=0.15, hspace=0.1)

        for i in range(n_plots):
            row = i // 2
            col = i % 2
            
            inner = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=outer[i], height_ratios=[6, 2], hspace=0.08
            )
            ax1 = fig.add_subplot(inner[0])
            ax2 = fig.add_subplot(inner[1], sharex=ax1)
            ax1.tick_params(labelbottom=False)

            # Force y-axis to use scientific notation
            formatter = plt.ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-1, 1))  # Force scientific notation for all values
            ax1.yaxis.set_major_formatter(formatter)

            tr_bar, tr_bin = np.histogram(true_list[i], bins=bins, range=ranges[i], weights=weights)
            pr_bar, _ = np.histogram(pred_list[i], bins=bins, range=ranges[i], weights=weights)

            hep.histplot(tr_bar, tr_bin, ax=ax1, lw=2, color="b", label=legend_lst[1])
            hep.histplot(pr_bar, tr_bin, ax=ax1, lw=2, color="r", label=legend_lst[0])
            ax1.set_xlim(ranges[i])
            
            if rmse_title:
                diff = np.array(pred_list[i]) - np.array(true_list[i])
                rmse = np.sqrt(np.mean(diff**2))
                sem = stats.sem(diff)
                ax1.set_title(f"{title[i]} (RMSE={rmse:.2f} ± {sem:.2f})", fontsize=title_size, loc="right")
            else:
                ax1.set_title(title[i], fontsize=title_size, loc="right")
                
            # Better legend positioning
            ax1.legend(fontsize=tick_size, loc='upper right')

            # Improved ratio calculation to avoid division issues
            epsilon = 1e-10
            tr_bar_safe = np.where(tr_bar == 0, epsilon, tr_bar)
            ratio = np.divide(pr_bar, tr_bar_safe)
            
            ax2.vlines(tr_bin[1:], 1, ratio, color="k", lw=1)
            for j, val in enumerate(ratio):
                if val > 2:
                    ax2.annotate(
                        "", xy=(tr_bin[j + 1], 2), xytext=(tr_bin[j + 1], 1.95),
                        arrowprops=dict(facecolor="k", shrink=0.05, width=1, headwidth=2),
                    )
                elif val < 0:
                    ax2.annotate(
                        "", xy=(tr_bin[j + 1], 0), xytext=(tr_bin[j + 1], 0.05),
                        arrowprops=dict(facecolor="k", shrink=0.05, width=1, headwidth=2),
                    )
                else:
                    ax2.scatter(tr_bin[j + 1], val, color="k", lw=1, s=10)

            ax2.set_ylim([0, 2])
            ax2.axhline(1, c="grey", ls="dashed", alpha=0.8)

            if col == 0:
                ax1.set_ylabel(ylabel[i], fontsize=label_size)
                ax2.set_ylabel(sub_ylabel[i], fontsize=label_size, loc="center", labelpad=10)
            else:
                ax1.set_ylabel("")
                ax2.set_ylabel("")
                ax1.tick_params(labelleft=True)
                ax2.tick_params(labelleft=False)

            # Only set x-labels on the bottom row (row == 3)
            if row == 3:
                ax2.set_xlabel(xlabel[i], fontsize=label_size, labelpad=xpad)
            else:
                ax2.set_xlabel("")
                ax2.tick_params(labelbottom=False)

            ax1.tick_params(axis="both", labelsize=tick_size)
            ax2.tick_params(axis="both", labelsize=tick_size, pad=10)

        # Handle empty subplot slots if n_plots < 8
        for i in range(n_plots, 8):
            row = i // 2
            col = i % 2
            ax = fig.add_subplot(outer[i])
            ax.axis('off')

        plt.subplots_adjust(left=0.12, right=0.92, top=0.95, bottom=0.08, wspace=0.25, hspace=0.1)

        if save_name:
            plt.savefig(f"{save_name}.png", dpi=dpi, bbox_inches="tight")
            print(f"Saved 1D grid as {save_name}.png")

        plt.show()
        plt.close()
    
    ########################
    # 2D Grid (4×2)
    ########################
    def hist_2d_grid_4plot(
        self,
        true_list,
        pred_list,
        ranges=(0.0, 1.0),
        xlabel="X",
        ylabel="Y",
        title="",
        bins=50,
        log=True,
        xpad=15,
        weights=None,
        save_name=None,
        dpi=500,
    ):
        """
        Plot a 4*2 grid of 2D histograms comparing true and predicted values with a shared color bar.
        """
        if not isinstance(ranges, list):
            ranges = [ranges] * 8
        if not isinstance(xlabel, list):
            xlabel = [xlabel] * 8
        if not isinstance(ylabel, list):
            ylabel = [ylabel] * 8
        if not isinstance(title, list):
            title = [title] * 8

        label_size = 20
        tick_size = 16
        title_size = 20

        fig = plt.figure(figsize=(14, 25), dpi=dpi)
        gs = gridspec.GridSpec(5, 2, height_ratios=[1, 1, 1, 1, 0.05], wspace=0.2, hspace=0.2)

        axes = []
        for i in range(8):
            row = i // 2
            col = i % 2
            ax = fig.add_subplot(gs[row, col])
            axes.append(ax)

        # Ensure ranges are valid
        for i in range(8):
            if not isinstance(ranges[i], (list, tuple)) or len(ranges[i]) != 2:
                ranges[i] = [min(true_list[i]), max(true_list[i])]

        # Find global max/min counts for consistent color scaling
        max_count = 0
        min_count = float('inf')
        for i in range(min(len(true_list), len(pred_list), 8)):
            hist_vals, _, _ = np.histogram2d(
                pred_list[i], true_list[i], bins=bins, range=[ranges[i], ranges[i]], weights=weights
            )
            max_count = max(max_count, np.max(hist_vals))
            non_zero_min = np.min(hist_vals[hist_vals > 0]) if np.any(hist_vals > 0) else 1
            min_count = min(min_count, non_zero_min)
        
        if min_count == float('inf'):
            min_count = 1  # Fallback for empty or all-zero data

        # Shared normalization
        if log:
            norm = LogNorm(vmin=min_count, vmax=max_count)
        else:
            norm = Normalize(vmin=min_count, vmax=max_count)

        # Create the plots with shared normalization
        for i, ax in enumerate(axes):
            row = i // 2
            col = i % 2
            
            formatter = plt.ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-1, 1))  # Force scientific notation for all values
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)
            
            h = ax.hist2d(
                pred_list[i], true_list[i], bins=bins, range=[ranges[i], ranges[i]],
                cmap="viridis", cmin=1, norm=norm, weights=weights,
            )

            ax.plot(ranges[i], ranges[i], "k--", alpha=0.7)
            r2_val = r2_score(true_list[i], pred_list[i])
            ax.set_title(rf"{title[i]} ($R^2$={r2_val:.2f})", fontsize=title_size, loc="right")

            if col == 0:
                ax.set_ylabel(ylabel[i], fontsize=label_size)
            else:
                ax.set_ylabel("")
                ax.tick_params(labelleft=True)

            if row == 3:  # Bottom row
                ax.set_xlabel(xlabel[i], fontsize=label_size, labelpad=xpad)
            else:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=True)  # Show ticks on all rows

            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(ranges[i])
            ax.set_ylim(ranges[i])
            ax.tick_params(axis="both", labelsize=tick_size, pad=8)
            ax.xaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
            ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))

        # Shared color bar at the bottom
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        cax = fig.add_subplot(gs[4, :])  
        cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.set_label("Counts" if log else "Counts", fontsize=label_size, labelpad=10)
        cbar.ax.tick_params(labelsize=tick_size)
        
        plt.subplots_adjust(left=0.12, right=0.92, top=0.95, bottom=0.06, wspace=0.12, hspace=0.1)

        if save_name:
            plt.savefig(f"{save_name}.png", dpi=dpi, bbox_inches="tight")
            print(f"Saved 2D grid as {save_name}.png")

        plt.show()
        plt.close()

    ########################
    # Combined Grid (2×3)
    ########################
    def hist_combined_grid(
        self,
        true_list_1d,
        pred_list_1d,
        true_list_2d,
        pred_list_2d,
        ranges=(-1, 1),
        xlabel="X",
        title="",
        row1_ylabel="Counts",
        row1_legend=["Separable", "SM"],
        row2_xlabel="True",
        row2_ylabel="Predicted",
        bins=50,
        log=True,
        xpad=10,
        rmse_title=False,
        weights=None,
        save_name=None,
        dpi=500,
    ):
        """
        Plot a 2x3 grid where first row has 1D histograms and second row has 2D histograms.
        """
        if not isinstance(ranges, list):
            ranges = [ranges] * 3
        if not isinstance(xlabel, list):
            xlabel = [xlabel] * 3
        if not isinstance(title, list):
            title = [title] * 3

        label_size = 20
        tick_size = 16
        title_size = 20

        fig = plt.figure(figsize=(18, 14), dpi=dpi)
        gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.18, hspace=0.1)

        # First row: 1D histograms with ratio plots
        first_row_axes = []
        for col in range(3):
            inner_gs = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=gs[0, col], height_ratios=[6, 2], hspace=0.08
            )
            ax1 = fig.add_subplot(inner_gs[0])
            ax2 = fig.add_subplot(inner_gs[1], sharex=ax1)
            first_row_axes.append((ax1, ax2))
            ax1.tick_params(labelbottom=False)

        # Second row: 2D histograms
        second_row_axes = []
        for col in range(3):
            ax = fig.add_subplot(gs[1, col])
            second_row_axes.append(ax)

        # Find global max/min counts for 2D histograms
        max_count = 0
        min_count = float('inf')
        for i in range(min(len(true_list_2d), len(pred_list_2d), 3)):
            hist_vals, _, _ = np.histogram2d(
                pred_list_2d[i], true_list_2d[i], bins=bins, range=[ranges[i], ranges[i]], weights=weights
            )
            max_count = max(max_count, np.max(hist_vals))
            non_zero_min = np.min(hist_vals[hist_vals > 0]) if np.any(hist_vals > 0) else 1
            min_count = min(min_count, non_zero_min)
        
        if min_count == float('inf'):
            min_count = 1

        if log:
            norm = LogNorm(vmin=min_count, vmax=max_count)
        else:
            norm = Normalize(vmin=min_count, vmax=max_count)

        print(f"Shared color bar range: vmin={min_count}, vmax={max_count}")  # Debug print

        # First row (1D histograms)
        for i, (ax1, ax2) in enumerate(first_row_axes):
            if i >= len(true_list_1d) or i >= len(pred_list_1d):
                continue
            
            formatter = plt.ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-1, 1))
            ax1.yaxis.set_major_formatter(formatter)

            tr_bar, tr_bin = np.histogram(true_list_1d[i], bins=bins, range=ranges[i], weights=weights)
            pr_bar, _ = np.histogram(pred_list_1d[i], bins=bins, range=ranges[i], weights=weights)

            hep.histplot(tr_bar, tr_bin, ax=ax1, lw=2, color="b", label=row1_legend[1])
            hep.histplot(pr_bar, tr_bin, ax=ax1, lw=2, color="r", label=row1_legend[0])
            ax1.set_xlim(ranges[i])

            if rmse_title is True:
                diff = np.array(pred_list_1d[i]) - np.array(true_list_1d[i])
                rmse = np.sqrt(np.mean(diff**2))
                sem = stats.sem(diff)
                ax1.set_title(f"{title[i]} (RMSE={rmse:.2f} ± {sem:.2f})", fontsize=title_size, loc="right")
            else:
                ax1.set_title(title[i], fontsize=title_size, loc="right")
            ax1.legend(fontsize=title_size)

            if i == 0:
                ax1.set_ylabel(row1_ylabel, fontsize=label_size)
            else:
                ax1.set_ylabel("")
                ax1.tick_params(labelleft=True)

            ratio = np.divide(pr_bar + 1, tr_bar + 1, where=(tr_bar != 0))
            ax2.vlines(tr_bin[1:], 1, ratio, color="k", lw=1)
            for j, val in enumerate(ratio):
                if val > 2:
                    ax2.annotate(
                        "", xy=(tr_bin[j + 1], 2), xytext=(tr_bin[j + 1], 1.95),
                        arrowprops=dict(facecolor="k", shrink=0.05, width=1, headwidth=2),
                    )
                elif val < 0:
                    ax2.annotate(
                        "", xy=(tr_bin[j + 1], 0), xytext=(tr_bin[j + 1], 0.05),
                        arrowprops=dict(facecolor="k", shrink=0.05, width=1, headwidth=2),
                    )
                else:
                    ax2.scatter(tr_bin[j + 1], val, color="k", lw=1, s=10)

            ax2.set_ylim([0, 2])
            ax2.axhline(1, c="grey", ls="dashed", alpha=0.8)
            ax2.set_xlabel(xlabel[i], fontsize=label_size, labelpad=xpad)

            if i == 0:
                ax2.set_ylabel(f"{row1_legend[0]}/{row1_legend[1]}", fontsize=label_size, loc="center", labelpad=10)
            else:
                ax2.set_ylabel("")
                ax2.tick_params(labelleft=True)

            ax1.tick_params(axis="both", labelsize=tick_size)
            ax2.tick_params(axis="both", labelsize=tick_size, pad=10)

        # Second row (2D histograms)
        for i, ax in enumerate(second_row_axes):
            if i >= len(true_list_2d) or i >= len(pred_list_2d):
                continue

            formatter = plt.ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-1, 1))
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)

            h = ax.hist2d(
                pred_list_2d[i], true_list_2d[i], bins=bins, range=[ranges[i], ranges[i]],
                cmap="viridis", cmin=1, norm=norm, weights=weights,
            )

            ax.plot(ranges[i], ranges[i], "k--", alpha=0.7)
            r2_val = r2_score(true_list_2d[i], pred_list_2d[i])
            ax.set_title(f"{title[i]} ($R^2$={r2_val:.2f})", fontsize=title_size, loc="right")

            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(ranges[i])
            ax.set_ylim(ranges[i])
            ax.tick_params(axis="both", labelsize=tick_size)

            if i == 0:
                ax.set_ylabel(row2_ylabel, fontsize=label_size)
            else:
                ax.set_ylabel("")
                ax.tick_params(labelleft=True)

            ax.set_xlabel(row2_xlabel, fontsize=label_size, labelpad=xpad)
            ax.xaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
            ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))

        # Shared color bar aligned with the last 2D subplot
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        cax = fig.add_subplot(gs[1, 3])  # Use GridSpec for initial placement
        last_ax_pos = second_row_axes[2].get_position()  # Get final position after layout

        # Manually shorten the color bar and center it vertically
        cbar_height_fraction = 0.8  # Shorten to 80% of the subplot height (adjust as needed)
        cbar_height = last_ax_pos.height * cbar_height_fraction
        cbar_y0 = last_ax_pos.y0 + (last_ax_pos.height - cbar_height) / 2  # Center vertically

        cax.set_position([
            last_ax_pos.x1 + 0.002,  # Move closer to the last subplot
            cbar_y0,                 # Adjusted y-position to center the shortened color bar
            0.02,                    # Width of the color bar
            cbar_height              # Shortened height
        ])
        cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
        cbar.set_label("Counts" if log else "Counts", fontsize=label_size, labelpad=10)
        cbar.ax.tick_params(labelsize=tick_size)

        plt.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.08, wspace=0.15, hspace=0.1)

        if save_name:
            plt.savefig(f"{save_name}.png", dpi=dpi, bbox_inches="tight")
            print(f"Saved combined grid as {save_name}.png")

        plt.show()
        plt.close()


    def hist_1d3plot(
        self, 
        true_list_1d,
        pred_list_1d,
        ranges=(-1, 1),
        xlabel="X",
        title="",
        row1_ylabel="Counts",
        row1_legend=["Separable", "SM"],
        bins=50,
        xpad=10,
        rmse_title=False,
        weights=None,
        save_name=None,
        dpi=500,
    ):
        """
        Plot a 1×3 grid with 1D histograms and ratio plots.
        """
        if not isinstance(ranges, list):
            ranges = [ranges] * 3
        if not isinstance(xlabel, list):
            xlabel = [xlabel] * 3
        if not isinstance(title, list):
            title = [title] * 3

        label_size = 20
        tick_size = 16
        title_size = 20

        # Change figure size to be wider than tall since we only have one row
        fig = plt.figure(figsize=(15, 8), dpi=dpi)
        
        # Create a single row grid with 3 columns
        gs = gridspec.GridSpec(1, 3, wspace=0.2)

        # Create histogram plots with ratio subplots
        first_row_axes = []
        for col in range(3):
            inner_gs = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=gs[0, col], height_ratios=[6, 2], hspace=0.08
            )
            ax1 = fig.add_subplot(inner_gs[0])
            ax2 = fig.add_subplot(inner_gs[1], sharex=ax1)
            first_row_axes.append((ax1, ax2))
            ax1.tick_params(labelbottom=False)

        # Plot 1D histograms
        for i, (ax1, ax2) in enumerate(first_row_axes):
            if i >= len(true_list_1d) or i >= len(pred_list_1d):
                continue
            
            formatter = plt.ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-1, 1))
            ax1.yaxis.set_major_formatter(formatter)

            tr_bar, tr_bin = np.histogram(true_list_1d[i], bins=bins, range=ranges[i], weights=weights)
            pr_bar, _ = np.histogram(pred_list_1d[i], bins=bins, range=ranges[i], weights=weights)

            hep.histplot(tr_bar, tr_bin, ax=ax1, lw=2, color="b", label=row1_legend[1])
            hep.histplot(pr_bar, tr_bin, ax=ax1, lw=2, color="r", label=row1_legend[0])
            ax1.set_xlim(ranges[i])

            if rmse_title is True:
                diff = np.array(pred_list_1d[i]) - np.array(true_list_1d[i])
                rmse = np.sqrt(np.mean(diff**2))
                sem = stats.sem(diff)
                ax1.set_title(f"{title[i]} (RMSE={rmse:.2f} ± {sem:.2f})", fontsize=title_size, loc="right")
            else:
                ax1.set_title(title[i], fontsize=title_size, loc="right")
            ax1.legend(fontsize=title_size)

            if i == 0:
                ax1.set_ylabel(row1_ylabel, fontsize=label_size)
            else:
                ax1.set_ylabel("")
                ax1.tick_params(labelleft=True)

            ratio = np.divide(pr_bar + 1, tr_bar + 1, where=(tr_bar != 0))
            ax2.vlines(tr_bin[1:], 1, ratio, color="k", lw=1)
            for j, val in enumerate(ratio):
                if val > 2:
                    ax2.annotate(
                        "", xy=(tr_bin[j + 1], 2), xytext=(tr_bin[j + 1], 1.95),
                        arrowprops=dict(facecolor="k", shrink=0.05, width=1, headwidth=2),
                    )
                elif val < 0:
                    ax2.annotate(
                        "", xy=(tr_bin[j + 1], 0), xytext=(tr_bin[j + 1], 0.05),
                        arrowprops=dict(facecolor="k", shrink=0.05, width=1, headwidth=2),
                    )
                else:
                    ax2.scatter(tr_bin[j + 1], val, color="k", lw=1, s=10)

            ax2.set_ylim([0, 2])
            ax2.axhline(1, c="grey", ls="dashed", alpha=0.8)
            ax2.set_xlabel(xlabel[i], fontsize=label_size, labelpad=xpad)

            if i == 0:
                ax2.set_ylabel(f"{row1_legend[0]}/{row1_legend[1]}", fontsize=label_size, loc="center", labelpad=10)
            else:
                ax2.set_ylabel("")
                ax2.tick_params(labelleft=True)

            ax1.tick_params(axis="both", labelsize=tick_size)
            ax2.tick_params(axis="both", labelsize=tick_size, pad=10)

        plt.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.12, wspace=0.15)

        if save_name:
            plt.savefig(f"{save_name}.png", dpi=dpi, bbox_inches="tight")
            print(f"Saved 1D histograms as {save_name}.png")

        plt.show()
        plt.close()

if __name__ == "__main__":
    plot = Plotter()

    # Test individual plots
    pred = np.random.normal(0, 1, 10000)
    truth = np.random.rand(10000)
    plot.hist_1d(pred, truth, ranges=(0, 1), xpad=25, save_name="1d_test")
    plot.hist_2d(pred, truth, ranges=(-2, 2), xpad=25, save_name="2d_test")

    # # Test grid 3 plots
    # true_list_1d = [np.random.uniform(-1, 1, 10000) for _ in range(6)]
    # pred_list_1d = [np.random.normal(0, 0.1, 10000) for _ in range(6)]
    # Test grid 4 plots
    true_list_1d = [np.random.uniform(-1, 1, 10000) for _ in range(8)]
    pred_list_1d = [np.random.normal(0, 0.1, 10000) for _ in range(8)]
    ranges_1d = (-1, 1)
    xlabels_1d = r"$\xi$ [None]"
    titles_1d = [
        r"$\xi^{(0)}_{n}$", r"$\xi^{(0)}_{r}$", r"$\xi^{(0)}_{k}$",
        r"$\xi^{(1)}_{n}$", r"$\xi^{(1)}_{r}$", r"$\xi^{(1)}_{k}$",
    ]
    plot.hist_1d_grid(
        true_list_1d, pred_list_1d, ranges=ranges_1d, xlabel=xlabels_1d,
        title=titles_1d, bins=30, save_name="1d_grid_test",
    )

    true_list_2d = [np.random.uniform(-1, 1, 10000) for _ in range(8)]
    pred_list_2d = [np.random.normal(0, 0.1, 10000) for _ in range(8)]
    ranges_2d = (-1, 1)
    xlabels_2d = r"True $\xi$ [None]"
    ylabels_2d = r"Predicted $\xi$ [None]"
    titles_2d = [
        r"$\xi^{(0)}_{n}$", r"$\xi^{(0)}_{r}$", r"$\xi^{(0)}_{k}$",
        r"$\xi^{(1)}_{n}$", r"$\xi^{(1)}_{r}$", r"$\xi^{(1)}_{k}$",
    ]
    plot.hist_2d_grid(
        true_list_2d, pred_list_2d, ranges=ranges_2d, xlabel=xlabels_2d,
        ylabel=ylabels_2d, title=titles_2d, bins=30, save_name="2d_grid_test",
    )

    plot.hist_combined_grid(
        true_list_1d, pred_list_1d, true_list_2d, pred_list_2d, ranges=(-1, 1),
        xlabel="X", title="", row1_ylabel="Counts", row1_legend=["Separable", "SM"],
        row2_xlabel="True", row2_ylabel="Predicted", bins=50, xpad=10,
        save_name="grid_test", dpi=300,
    )
    
    plot.hist_1d3plot(
        true_list_1d, pred_list_1d, ranges=(-1, 1), xlabel="X", title="",
        row1_ylabel="Counts", row1_legend=["Separable", "SM"], bins=50, xpad=10,
        save_name="1d3plot_test", dpi=300,
    )
    
    plot.hist_1d_grid_4plots(
        true_list_1d, pred_list_1d, ranges=(-1, 1), xlabel="X", title="",
        ylabel="Counts", legend_lst=["Pred", "True"], bins=50, xpad=10,
        rmse_title=False, weights=None, save_name="1d_grid_4plots_test", dpi=300,
    )
    plot.hist_2d_grid_4plot(
        true_list_2d, pred_list_2d, ranges=(-1, 1), xlabel="X", ylabel="Y",
        title="", bins=50, log=True, xpad=10, weights=None,
        save_name="2d_grid_4plot_test", dpi=300,
    )

    print("Done!")