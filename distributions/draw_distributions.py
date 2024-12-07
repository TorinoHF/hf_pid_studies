import argparse
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from flarefly.data_handler import DataHandler
from flarefly.fitter import F2MassFitter
import yaml
import ROOT

ROOT.gROOT.SetBatch(True)

# Custom function for the lower triangle: heatmap
def lower_triangle_heatmap(x, y, **kwargs):
    data = kwargs.pop('data', None)
    if x.name != y.name:  # No diagonal elements
        # Plot correlation coefficient
        corr = data.corr()
        corr_value = corr.loc[y.name, x.name]
        sns.heatmap(
            pd.DataFrame([[corr_value]]),  # Single value as 1x1 heatmap
            vmin=-1, vmax=1, annot=True, cbar=False, square=True, cmap="coolwarm",
            xticklabels=False, yticklabels=False, ax=plt.gca()
        )

# Custom function for the upper triangle: scatter plot
def upper_triangle_scatter(x, y, **kwargs):
    if x.name != y.name:  # Avoid diagonal elements
        sns.scatterplot(x=x, y=y, **kwargs)

def draw_correlation_pt(data_df, suffix, pt_min, pt_max, cfg):
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    # Create PairGrid
    g = sns.PairGrid(data_df, vars=[*cfg['dau_pt_var_names'], cfg['mother_pt_var_name']])

    # Map functions to the lower and upper triangle
    g.map_lower(lower_triangle_heatmap, cmap="coolwarm", data=data_df)
    g.map_upper(sns.histplot, bins=40)
    g.map_diag(sns.histplot, bins=40)

    # Adjust layout
    plt.tight_layout()

    output_dir = os.path.join(cfg['output']['dir'], f'{pt_min*10:.0f}_{pt_max*10:.0f}')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    out_path = os.path.join(
        output_dir,
        f'corr_matrix_{suffix}.png'
    )
    g.fig.savefig(out_path, dpi=300, bbox_inches="tight")


def fit_mass(df, suffix, pt_min, pt_max, cfg):
    # Create the data handler
    data_handler = DataHandler(df, cfg["mother_mass_var_name"])
    fitter = F2MassFitter(data_handler, ["doublecb"], ["expo"])
    fitter.set_signal_initpar(0, "mu", 0.5)
    fitter.set_signal_initpar(0, "sigma", 0.01)
    fitter.set_background_initpar(0, "lam", -0.1)

    # Fit the data
    fitter.mass_zfit()

    loc = ["lower left", "upper left"]
    ax_title = r"$M(\mathrm{\pi\pi})$ GeV$/c^2$"
    fig, _ = fitter.plot_mass_fit(
        style="ATLAS",
        show_extra_info = fitter.get_background()[1] != 0,
        figsize=(8, 8), extra_info_loc=loc,
        axis_title=ax_title
    )

    output_dir = os.path.join(cfg['output']['dir'], f'{pt_min*10:.0f}_{pt_max*10:.0f}')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig.savefig(
        os.path.join(
            output_dir,
            f'mass_fit_{suffix}.png'
        ),
        dpi=300, bbox_inches="tight"
    )
    return fitter

def draw_pid_distributions(dfs, cfg, labels, weights, pt_min, pt_max):
    for var in cfg['variables_to_plot']:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        for df, label, weight in zip(dfs, labels, weights):
            df[var].hist(bins=100, label=label, weights=weight, alpha=0.5, density=True, range=(-5,5))
        ax.set_xlabel(var)
        ax.set_ylabel('Entries')
        ax.legend()
        output_dir = os.path.join(cfg['output']['dir'], f'{pt_min*10:.0f}_{pt_max*10:.0f}')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fig.savefig(
            os.path.join(
                output_dir,
                f'{var}.png'
            ),
            dpi=300, bbox_inches="tight"
        )

def draw_efficiencies(dfs, cfg, labels, pt_bins):
    for var, pt_var in zip(cfg['variables_to_plot'], cfg['dau_pt_var_names']):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
        min_eff = 1
        max_eff = 0
        min_ratio = 1000
        max_ratio = 0
        for nsigma in [3, 2, 1]:
            effs = []
            effs_unc = []
            for df, label in zip(dfs, labels):
                effs.append([])
                effs_unc.append([])
                for pt_min, pt_max in zip(pt_bins[:-1], pt_bins[1:]):
                    if "Tpc" not in var:
                        n_sel = len(df.query(f'abs({var}) < {nsigma} and {pt_min} < {pt_var} < {pt_max}'))
                    else:
                        n_sel = len(df.query(f'(abs({var}) < {nsigma} or {var}==-999) and {pt_min} < {pt_var} < {pt_max}'))
                    n_total = len(df.query(f'{pt_min} < {pt_var} < {pt_max}'))
                    eff = n_sel / n_total if n_total > 0 else 0
                    eff_unc = np.sqrt(eff * (1 - eff) / n_total) if n_total > 0 else 0
                    effs[-1].append(eff)
                    effs_unc[-1].append(eff_unc)
                pt_centers = (np.array(pt_bins[:-1]) + np.array(pt_bins[1:])) / 2
                pt_widths = (np.array(pt_bins[1:]) - np.array(pt_bins[:-1])) / 2
                ax1.errorbar(pt_centers, effs[-1], yerr=effs_unc[-1], xerr=pt_widths, label=f'{label}, |{var}| < {nsigma}', fmt='o')

            ratio = np.array(effs[0]) / np.array(effs[1])
            ratio_unc = ratio * np.sqrt((np.array(effs_unc[0]) / np.array(effs[0]))**2 + (np.array(effs_unc[1]) / np.array(effs[1]))**2)
            ax2.errorbar(pt_centers, ratio, yerr=ratio_unc, xerr=pt_widths, label=f'|{var}| < {nsigma}', fmt='o')

            if np.array(effs).flatten()[np.array(effs).flatten()!=0].min() < min_eff:
                min_eff = np.array(effs).flatten()[np.array(effs).flatten()!=0].min()
            if np.array(effs).flatten().max() > max_eff:
                max_eff = np.array(effs).flatten().max()
            if ratio[~np.isnan(ratio)].min() < min_ratio:
                min_ratio = ratio[~np.isnan(ratio)].min()
            if ratio[~np.isnan(ratio)].max() > max_ratio:
                max_ratio = ratio[~np.isnan(ratio)].max()

        ax1.set_ylabel('Efficiency')
        ax1.set_ylim(min_eff.min()/2, 1.2)
        ax1.legend()
        ax1.set_yscale('log')
        ax1.grid(True)

        ax2.set_xlabel(r'$p_T$ (GeV/$c$)')
        ax2.set_ylabel('Data / MC')
        ax2.set_ylim(min_ratio*0.95, max_ratio*1.05)
        ax2.legend()
        ax2.grid(True)

        output_dir = os.path.join(cfg['output']['dir'], 'efficiencies')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fig.savefig(os.path.join(output_dir, f'{var}.png'), dpi=300, bbox_inches="tight")
        plt.close(fig)

def draw_distributions(cfg_file_name):
    # Read the configuration file
    with open(cfg_file_name, 'r') as cfg_file:
        cfg = yaml.safe_load(cfg_file)

    pt_bins = cfg["pt_bins"]
    
    # Read the input data
    data_df = pd.read_parquet(cfg['inputs']['data'])
    mc_df = pd.read_parquet(cfg['inputs']['mc'])
    for pt_min, pt_max in zip(pt_bins[:-1], pt_bins[1:]):
        selection = f"{cfg['mass_range'][0]} < {cfg['mother_mass_var_name']} < {cfg['mass_range'][1]} and "
        for axis_pt in cfg['dau_pt_var_names']:
            selection += f'{pt_min} < {axis_pt} < {pt_max} and '
        selection = selection[:-5]
        df_data_pt = data_df.query(selection)
        df_mc_pt = mc_df.query(selection)

        draw_correlation_pt(df_data_pt, 'data', pt_min, pt_max, cfg)
        draw_correlation_pt(df_mc_pt, 'mc', pt_min, pt_max, cfg)
        fitter = fit_mass(df_data_pt, 'data', pt_min, pt_max, cfg)
        if not np.isclose(fitter.get_background()[0], 1, atol=1):
            draw_pid_distributions([df_data_pt, df_mc_pt], cfg, ['data', 'mc'], [fitter.get_sweights()['signal'], None], pt_min, pt_max)
        else:
            draw_pid_distributions([df_data_pt, df_mc_pt], cfg, ['data', 'mc'], [None, None], pt_min, pt_max)
    draw_efficiencies([data_df, mc_df], cfg, ['data', 'mc'], pt_bins)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Draw distributions')
    parser.add_argument('config_file', help='Path to the input configuration file')
    args = parser.parse_args()

    draw_distributions(args.config_file)