import re
import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

def convert_to_interval(interval_str):
    match = re.match(r"(\[|\()(\d+(\.\d+)?),\s*(\d+(\.\d+)?)(\]|\))", interval_str)
    if match:
        # Extract the left and right bounds, converting them to float
        left = float(match.group(2))  # Group 2 captures the left bound
        right = float(match.group(4))  # Group 4 captures the right bound
        
        # Determine if the interval is closed on the left or right
        closed = 'left' if match.group(1) == '[' else 'right'
        # Return the interval as a pandas Interval
        return pd.Interval(left, right, closed=closed)
    
    # If the string doesn't match the interval format, return None
    return None  

def sort_df_by_interval(df, col):
    # Use .loc[] to safely modify the DataFrame and avoid the SettingWithCopyWarning
    df.loc[:, "lower_bound"] = df[col].apply(lambda x: x.left)  # Extract left bound
    df_sorted = df.sort_values("lower_bound").drop(columns="lower_bound")  # Sort and drop helper column
    return df_sorted

def get_label(diff_col_name):
    if diff_col_name == "fPt":
        return '$p_T$ (GeV/$c$)'
    if diff_col_name == "fOccupancyFt0c":
        return 'Occupancy FTOC (arb. units)'
    if diff_col_name == "fCentralityFT0C" or diff_col_name == "fCentralityFT0M":
        return 'Centrality'

def draw_efficiencies(dfs_data, dfs_mc, diff_col_name, eff_var, labels):

    cmap = plt.get_cmap('tab20')
    sort_dfs_data, sort_dfs_mc = [], []
    for df_data, df_mc in zip(dfs_data, dfs_mc):
        sort_dfs_data.append(sort_df_by_interval(df_data, diff_col_name))
        sort_dfs_mc.append(sort_df_by_interval(df_mc, diff_col_name))
    
    fig, axs = plt.subplots(3, 2, figsize=(18, 18))

    lower_bin_bounds = sort_dfs_data[0][diff_col_name].apply(lambda x: x.left)
    upper_bin_bounds = sort_dfs_data[0][diff_col_name].apply(lambda x: x.right)
    bin_centers = (np.array(lower_bin_bounds) + np.array(upper_bin_bounds)) / 2
    bin_widths = (np.array(upper_bin_bounds) - np.array(lower_bin_bounds)) / 2
    ticks = list(lower_bin_bounds) + [list(upper_bin_bounds)[-1]]
    for ax in axs.flat:
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks, rotation=45, ha='right')

    for i_df, (df_data, df_mc, label) in enumerate(zip(sort_dfs_data, sort_dfs_mc, labels)):
        
        # data and MC leg entries side by side in overlap plots
        data_marker = Line2D([0], [0], color=cmap(i_df*2), marker='o', linestyle='None')
        mc_marker = Line2D([0], [0], color=cmap(i_df*2+1), marker='o', linestyle='None')
        if 'legend_entries' not in locals():
            legend_entries = []
        legend_entries.append(((data_marker, mc_marker), f'{label} data (mc)'))
        
        n_sigma_vals = [3,2,1]
        for i_sigma, n_sigma in enumerate(n_sigma_vals):
            n_sigma_effs_data, n_sigma_effs_uncs_data, n_sigma_effs_mc, n_sigma_effs_uncs_mc = [], [], [], []
            for entry in range(len(df_data)):
                n_sigma_effs_data.append(df_data.loc[entry, eff_var][i_sigma])
                n_sigma_effs_uncs_data.append(df_data.loc[entry, f"{eff_var}_unc"][i_sigma])
                n_sigma_effs_mc.append(df_mc.loc[entry, eff_var][i_sigma])
                n_sigma_effs_uncs_mc.append(df_mc.loc[entry, f"{eff_var}_unc"][i_sigma])

            axs[i_sigma, 0].errorbar(bin_centers, n_sigma_effs_data, c=cmap(i_df*2), 
                                     xerr=bin_widths, fmt='p')
            axs[i_sigma, 0].errorbar(bin_centers, n_sigma_effs_mc, c=cmap(i_df*2+1), 
                                     xerr=bin_widths, fmt='p')
            detector = 'TPC' if 'Tpc' in eff_var else 'TOF'
            var_for_title = r'N$\sigma^\pi_{' + detector + '}$'
            axs[i_sigma, 0].set_title(rf"|{var_for_title}| < {n_sigma}")
            axs[i_sigma, 0].set_ylabel('Efficiency')
            # axs[i_sigma, 0].set_ylim(min_eff.min()/2, 1.2)
            axs[i_sigma, 0].set_xlabel(get_label(diff_col_name))
            axs[i_sigma, 0].set_yscale('log')
            axs[i_sigma, 0].grid(True)

            custom_legend = []
            for markers, lbl in legend_entries:
                custom_legend.append((markers, lbl))

            # Add legend to the plot with HandlerTuple for side-by-side markers
            axs[i_sigma, 0].legend([entry[0] for entry in custom_legend], 
                             [entry[1] for entry in custom_legend], 
                             handler_map={tuple: HandlerTuple(ndivide=None)}, loc='best')
        
            ratio = np.array(n_sigma_effs_data) / np.array(n_sigma_effs_mc)
            ratio_unc = ratio * np.sqrt((np.array(n_sigma_effs_uncs_data) / np.array(n_sigma_effs_data))**2 + (np.array(n_sigma_effs_uncs_mc) / np.array(n_sigma_effs_mc))**2)

            axs[i_sigma, 1].errorbar(bin_centers, ratio, yerr=ratio_unc, xerr=bin_widths, label=label, fmt='o', color=cmap(i_df*2))

            axs[i_sigma, 1].set_title(rf"Ratio |{var_for_title}| < {n_sigma}")
            axs[i_sigma, 1].set_xlabel(get_label(diff_col_name))
            axs[i_sigma, 1].set_ylabel('Data / MC')
            # axs[i_sigma, 1].set_ylim(min_ratio*0.95, max_ratio*1.05)
            axs[i_sigma, 1].legend()
            axs[i_sigma, 1].grid(True)
            axs[i_sigma, 1].axhline(y=1, color='k', linestyle='--')

    return fig

def draw_plots(input_folder):
    df_neg_pi_mc = pd.read_parquet(f"{input_folder}/neg_pi_eff_df_mc.parquet", engine="pyarrow") 
    df_pos_pi_mc = pd.read_parquet(f"{input_folder}/pos_pi_eff_df_mc.parquet", engine="pyarrow") 
    df_neg_pi = pd.read_parquet(f"{input_folder}/neg_pi_eff_df.parquet", engine="pyarrow") 
    df_pos_pi = pd.read_parquet(f"{input_folder}/pos_pi_eff_df.parquet", engine="pyarrow")

    interval_cols = ['fCentralityFT0C', 'fPt']

    for col in interval_cols:
        df_neg_pi_mc[col] = df_neg_pi_mc[col].apply(convert_to_interval) 
        df_pos_pi_mc[col] = df_pos_pi_mc[col].apply(convert_to_interval) 
        df_neg_pi[col] = df_neg_pi[col].apply(convert_to_interval) 
        df_pos_pi[col] = df_pos_pi[col].apply(convert_to_interval)

    class_var = 'fCentralityFT0C'
    class_var_name = 'cent'
    bins = [0, 10, 20, 30, 40, 50, 60, 80, 100]
    labels = [f'{min} < {class_var_name} < {max}' for min, max in zip(bins[:-1], bins[1:])]

    dfs_pos_pi_mc = [df_pos_pi_mc.query(f'{class_var} == @pd.Interval({min}, {max}, closed="left")').reset_index(drop=True) for min, max in zip(bins[:-1], bins[1:])]
    dfs_pos_pi = [df_pos_pi.query(f'{class_var} == @pd.Interval({min}, {max}, closed="left")').reset_index(drop=True) for min, max in zip(bins[:-1], bins[1:])]
    dfs_neg_pi_mc = [df_neg_pi_mc.query(f'{class_var} == @pd.Interval({min}, {max}, closed="left")').reset_index(drop=True) for min, max in zip(bins[:-1], bins[1:])]
    dfs_neg_pi = [df_neg_pi.query(f'{class_var} == @pd.Interval({min}, {max}, closed="left")').reset_index(drop=True) for min, max in zip(bins[:-1], bins[1:])]
    
    if not os.path.exists(f"{input_folder}/figures/efficiencies"):
        os.makedirs(f"{input_folder}/figures/efficiencies")

    diff_var = 'fPt'
    fig = draw_efficiencies(dfs_neg_pi, dfs_neg_pi_mc, diff_var, 'fNSigmaTpcNegPi', labels)
    fig.savefig(f"{input_folder}/figures/efficiencies/tpc_neg_pi_pt.png", bbox_inches='tight')
    fig = draw_efficiencies(dfs_pos_pi, dfs_pos_pi_mc, diff_var, 'fNSigmaTpcPosPi', labels)
    fig.savefig(f"{input_folder}/figures/efficiencies/tpc_pos_pi_pt.png", bbox_inches='tight')
    fig = draw_efficiencies(dfs_neg_pi, dfs_neg_pi_mc, diff_var, 'fNSigmaTofNegPi', labels)
    fig.savefig(f"{input_folder}/figures/efficiencies/tof_neg_pi_pt.png", bbox_inches='tight')
    fig = draw_efficiencies(dfs_pos_pi, dfs_pos_pi_mc, diff_var, 'fNSigmaTofPosPi', labels)
    fig.savefig(f"{input_folder}/figures/efficiencies/tof_pos_pi_pt.png", bbox_inches='tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('input_folder', help='Input folder')
    args = parser.parse_args()

    draw_plots(args.input_folder)