import argparse
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from flarefly.data_handler import DataHandler
from flarefly.fitter import F2MassFitter
import yaml
import ROOT
from draw_correlations import draw_correlation_pt

ROOT.gROOT.SetBatch(True)

def fit_mass(df, suffix, pt_min, pt_max, cfg, sub_dir):
    # Create the data handler
    data_handler = DataHandler(df, cfg["mother_mass_var_name"])
    fitter = F2MassFitter(data_handler, ["doublecb"], ["expo"])
    fitter.set_signal_initpar(0, "mu", cfg["mean"])
    fitter.set_signal_initpar(0, "sigma", cfg["sigma"])
    fitter.set_background_initpar(0, "lam", -0.1)

    # Fit the data
    fitter.mass_zfit()

    loc = ["lower left", "upper left"]
    if cfg["mother_mass_var_name"] == "fMassK0":
        ax_title = r"$M(\mathrm{\pi\pi})$ GeV$/c^2$"
    elif cfg["mother_mass_var_name"] == "fMassOmega":
        ax_title = r"$M(\mathrm{\Lambda\text{K}})$ GeV$/c^2$"
    elif cfg["mother_mass_var_name"] == "fMassLambda":
        ax_title = r"$M(\mathrm{p\pi})$ GeV$/c^2$"
    elif cfg["mother_mass_var_name"] == "fMassAntiLambda":
        ax_title = r"$M(\mathrm{\overline{p}\pi})$ GeV$/c^2$"
    else:
        ax_title = "Unknown mass variable"

    fig, _ = fitter.plot_mass_fit(
        style="ATLAS",
        show_extra_info = fitter.get_background()[1] != 0,
        figsize=(8, 8), extra_info_loc=loc,
        axis_title=ax_title
    )

    output_dir = os.path.join(cfg['output']['dir'], f'{sub_dir}/{pt_min*10:.0f}_{pt_max*10:.0f}')
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

def draw_pid_distributions(dfs, cfg, labels, weights, pt_min, pt_max, sub_dir):
    for var in cfg['variables_to_plot']:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        for df, label, weight in zip(dfs, labels, weights):
            df[var].hist(bins=100, label=label, alpha=0.5, density=True, range=(-5,5))
        ax.set_xlabel(var)
        ax.set_ylabel('Entries')
        ax.legend()
        output_dir = os.path.join(cfg['output']['dir'], f'{sub_dir}/{pt_min*10:.0f}_{pt_max*10:.0f}')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fig.savefig(
            os.path.join(
                output_dir,
                f'{var}.png'
            ),
            dpi=300, bbox_inches="tight"
        )

def get_efficiency(dfs, var):
    effs = []
    effs_unc = []
    for df in dfs:
        effs.append([])
        effs_unc.append([])
        for nsigma in [3, 2, 1]:
            # print(f'NSIGMA {nsigma}')
            if "Tpc" not in var:
                n_sel = len(df.query(f'abs({var}) < {nsigma}'))
            else:
                n_sel = len(df.query(f'(abs({var}) < {nsigma} or {var}==-999)'))
            n_total = len(df)
            eff = n_sel / n_total if n_total > 0 else 0
            eff_unc = np.sqrt(eff * (1 - eff) / n_total) if n_total > 0 else 0
            effs[-1].append(eff)
            effs_unc[-1].append(eff_unc)
    
    # for eff_var in effs:
    #     print(eff_var)
    # print(f"effs.size(): {len(effs)}")
    # print(f"effs[0].size(): {len(effs[0])}")
    # print(f"effs_unc.size(): {len(effs_unc)}")
    return effs, effs_unc

# def draw_efficiencies(dfs, cfg, labels, pt_bins, dau_name):
#     for var, pt_var in zip(cfg['variables_to_plot'], cfg['dau_pt_var_names']):
#         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
#         min_eff = 1
#         max_eff = 0
#         min_ratio = 1000
#         max_ratio = 0
#         for nsigma in [3, 2, 1]:
#             effs = []
#             effs_unc = []
#             for df, label in zip(dfs, labels):
#                 effs.append([])
#                 effs_unc.append([])
#                 for pt_min, pt_max in zip(pt_bins[:-1], pt_bins[1:]):
#                     if "Tpc" not in var:
#                         n_sel = len(df.query(f'abs({var}) < {nsigma} and {pt_min} < {pt_var} < {pt_max}'))
#                     else:
#                         n_sel = len(df.query(f'(abs({var}) < {nsigma} or {var}==-999) and {pt_min} < {pt_var} < {pt_max}'))
#                     n_total = len(df.query(f'{pt_min} < {pt_var} < {pt_max}'))
#                     eff = n_sel / n_total if n_total > 0 else 0
#                     eff_unc = np.sqrt(eff * (1 - eff) / n_total) if n_total > 0 else 0
#                     effs[-1].append(eff)
#                     effs_unc[-1].append(eff_unc)
#                 pt_centers = (np.array(pt_bins[:-1]) + np.array(pt_bins[1:])) / 2
#                 pt_widths = (np.array(pt_bins[1:]) - np.array(pt_bins[:-1])) / 2
#                 ax1.errorbar(pt_centers, effs[-1], yerr=effs_unc[-1], xerr=pt_widths, label=f'{label}, |{var}| < {nsigma}', fmt='o')

#             ratio = np.array(effs[0]) / np.array(effs[1])
#             ratio_unc = ratio * np.sqrt((np.array(effs_unc[0]) / np.array(effs[0]))**2 + (np.array(effs_unc[1]) / np.array(effs[1]))**2)
#             ax2.errorbar(pt_centers, ratio, yerr=ratio_unc, xerr=pt_widths, label=f'|{var}| < {nsigma}', fmt='o')

#             if np.array(effs).flatten()[np.array(effs).flatten()!=0].min() < min_eff:
#                 min_eff = np.array(effs).flatten()[np.array(effs).flatten()!=0].min()
#             if np.array(effs).flatten().max() > max_eff:
#                 max_eff = np.array(effs).flatten().max()
#             if ratio[~np.isnan(ratio)].min() < min_ratio:
#                 min_ratio = ratio[~np.isnan(ratio)].min()
#             if ratio[~np.isnan(ratio)].max() > max_ratio:
#                 max_ratio = ratio[~np.isnan(ratio)].max()

#         ax1.set_ylabel('Efficiency')
#         ax1.set_ylim(min_eff.min()/2, 1.2)
#         ax1.legend()
#         ax1.set_yscale('log')
#         ax1.grid(True)

#         ax2.set_xlabel(r'$p_T$ (GeV/$c$)')
#         ax2.set_ylabel('Data / MC')
#         ax2.set_ylim(min_ratio*0.95, max_ratio*1.05)
#         ax2.legend()
#         ax2.grid(True)

#         output_dir = os.path.join(cfg['output']['dir'], f'{dau_name}/efficiencies')
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)
#         fig.savefig(os.path.join(output_dir, f'{var}.png'), dpi=300, bbox_inches="tight")
#         plt.close(fig)

def get_selections(cfg):
    selection_vars = []
    selection_vars_ranges = []
    if cfg.get('cent_bins'):
        selection_vars.append("fCentralityFT0C")
        selection_vars_ranges.append(cfg["cent_bins"])
    if cfg.get('occ_bins'):
        selection_vars.append("fOccupancyFt0c")
        selection_vars_ranges.append(cfg["occ_bins"])
    
    # obtain all possible combinations of event selections
    ev_sels = list(itertools.product(*selection_vars_ranges))
    
    return selection_vars, ev_sels

def draw_distributions(cfg_file_name):
    # Read the configuration file
    with open(cfg_file_name, 'r') as cfg_file:
        cfg = yaml.safe_load(cfg_file)

    pt_bins = cfg["pt_bins"]
    selection_vars, ev_sels = get_selections(cfg)

    eff_df_cols = selection_vars + ["fPt"] + [
        item for pair in zip(cfg["variables_to_plot"], [f"{var}_unc" for var in cfg["variables_to_plot"]]) for item in pair
    ]

    eff_dfs = []
    eff_dfs_mc = []
    for dau in cfg['dau_names']:
        eff_dfs.append(pd.DataFrame(columns=eff_df_cols))
        eff_dfs_mc.append(pd.DataFrame(columns=eff_df_cols))

    # Read the input data
    data_df = pd.read_parquet(cfg['inputs']['data'])
    mc_df = pd.read_parquet(cfg['inputs']['mc'])

    for sel in ev_sels:
        selection = f"{cfg['mass_range'][0]} < {cfg['mother_mass_var_name']} < {cfg['mass_range'][1]} and "
        out_dir = ""
        eff_df_sel_row = []
        for var_name, range in zip(selection_vars, sel):
            selection += f'{range[0]} < {var_name} < {range[1]} and '
            # eff_df_sel_row.append(pd.Interval(range[0], range[1], closed='left')) 
            eff_df_sel_row.append(f"[{range[0]}, {range[1]})") 
            out_dir += f"{var_name}_{range[0]}_{range[1]}/" 
        
        for pt_min, pt_max in zip(pt_bins[:-1], pt_bins[1:]):
            for dau, dau_axis_pt, dau_df, dau_df_mc in zip(cfg['dau_names'], cfg['dau_pt_var_names'], eff_dfs, eff_dfs_mc):
                out_daudir = f"{dau}/" + out_dir 
                dau_pt_sel = selection + f'{pt_min} < {dau_axis_pt} < {pt_max}'
                
                df_data_pt = data_df.query(dau_pt_sel)
                df_mc_pt = mc_df.query(dau_pt_sel)

                fitter = fit_mass(df_data_pt, 'data', pt_min, pt_max, cfg, out_daudir)

                # print('PRE EFF EVAL')
                # eff_df_row = [*eff_df_sel_row] + [pd.Interval(pt_min, pt_max, closed='left')]
                # eff_df_mc_row = [*eff_df_sel_row] + [pd.Interval(pt_min, pt_max, closed='left')]
                eff_df_row = [*eff_df_sel_row] + [f"[{pt_min}, {pt_max})"]
                eff_df_mc_row = [*eff_df_sel_row] + [f"[{pt_min}, {pt_max})"]
                for var in cfg["variables_to_plot"]:
                    effs, effs_uncs = get_efficiency([df_data_pt, df_mc_pt], var)
                    eff_df_row = eff_df_row + [effs[0]] + [effs_uncs[0]]
                    eff_df_mc_row = eff_df_mc_row + [effs[1]] + [effs_uncs[1]]

                # print('AFTER EFF EVAL')
                # print(f"eff_df_row: {eff_df_row}")
                # print(f"eff_df_mc_row: {eff_df_mc_row}")
                # effs, effs_uncs = get_efficiency([df_data_pt, df_mc_pt], cfg["variables_to_plot"])
                # eff_df_row = [*eff_df_sel_row, pd.Interval(pt_min, pt_max, closed='left'), *effs[0], *effs_uncs[0]]
                dau_df.loc[len(dau_df)] = eff_df_row
                # eff_df_mc_row = [*eff_df_sel_row, pd.Interval(pt_min, pt_max, closed='left'), *effs[1], *effs_uncs[1]]
                dau_df_mc.loc[len(dau_df_mc)] = eff_df_mc_row
                # print(f"eff_df_row: {eff_df_row}")
                # print(f"eff_df_mc_row: {eff_df_mc_row}")
                # print('END')
                
                if cfg.get('draw_corr'):
                    draw_correlation_pt(df_data_pt, 'data', pt_min, pt_max, cfg, out_daudir)
                    draw_correlation_pt(df_mc_pt, 'mc', pt_min, pt_max, cfg, out_daudir)
                if not np.isclose(fitter.get_background()[0], 1, atol=1):
                    draw_pid_distributions([df_data_pt, df_mc_pt], cfg, ['data', 'mc'], [fitter.get_sweights()['signal'], None], pt_min, pt_max, out_daudir)
                else:
                    draw_pid_distributions([df_data_pt, df_mc_pt], cfg, ['data', 'mc'], [None, None], pt_min, pt_max, out_daudir)
            # draw_efficiencies([data_df, mc_df], cfg, ['data', 'mc'], pt_bins, out_daudir)
    
    for dau, eff_df, eff_df_mc in zip(cfg['dau_names'], eff_dfs, eff_dfs_mc):
        # print('\n\n\n')
        # print(eff_df)
        # print('\n')
        # print(eff_df_mc)
        eff_df.to_parquet(f"{cfg["output"]["dir"]}/{dau}_eff_df.parquet") #, engine='fastparquet')
        eff_df_mc.to_parquet(f"{cfg["output"]["dir"]}/{dau}_eff_df_mc.parquet") #, engine='fastparquet')
        # print('\n\n\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Draw distributions')
    parser.add_argument('config_file', help='Path to the input configuration file')
    args = parser.parse_args()

    draw_distributions(args.config_file)