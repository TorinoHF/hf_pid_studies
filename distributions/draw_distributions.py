import argparse
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # pylint: disable=wrong-import-position
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import itertools
from flarefly.data_handler import DataHandler
from flarefly.fitter import F2MassFitter
import yaml
import ROOT
from draw_correlations import draw_correlation_pt

ROOT.gROOT.SetBatch(True)

def get_distribution_mean_sigma(df, var):
    df = df.query(f"abs({var}) < 5")
    if len(df) == 0:
        print(f"No dataframe entries left when cutting abs({var}) < 5")
        return 0., 0.
    if "w_splot" in df.columns and sum(df['w_splot'])>0:
        mean = np.average(df[var], weights=df['w_splot'])
        sigma = np.sqrt(np.average((df[var] - mean)**2, weights=df['w_splot']))
    else:
        mean = np.average(df[var])
        sigma = df[var].std()
    return mean, sigma

def fit_mass(df, suffix, pt_min, pt_max, sel, cfg, sub_dir, sel_vars):
    # Create the data handler
    data_handler = DataHandler(df, cfg["mother_mass_var_name"])
    sgn_func = [cfg["fit_config"]["sgn_func"]] if cfg["fit_config"].get('sgn_func') else ["doublecb"]
    bkg_func = [cfg["fit_config"]["bkg_func"]] if cfg["fit_config"].get('bkg_func') else ["nobkg"]
    fitter_name = f"{sub_dir.split('/')[0]}_{suffix}_pt_{pt_min*10:.0f}_{pt_max*10:.0f}"
    for range, range_name in zip(sel, sel_vars):
        fitter_name += f"_{range_name}_{range[0]}_{range[1]}"
        
    try:
        fitter = F2MassFitter(data_handler, sgn_func, bkg_func, verbosity=0, name=fitter_name)
        fitter.set_signal_initpar(0, "mu", cfg["fit_config"]["mean"])
        fitter.set_signal_initpar(0, "sigma", cfg["fit_config"]["sigma"])
        if sgn_func[0] == "doublecb":
            fitter.set_signal_initpar(0, "alphal", cfg["fit_config"]["alphal"], limits=[0.5, 10])
            fitter.set_signal_initpar(0, "alphar", cfg["fit_config"]["alphar"], limits=[0.5, 10])
            fitter.set_signal_initpar(0, "nl", cfg["fit_config"]["nl"], limits=[0.5, 30])
            fitter.set_signal_initpar(0, "nr", cfg["fit_config"]["nr"], limits=[0.5, 30])
        if bkg_func[0] == "chebpol1":
            fitter.set_background_initpar(0, "c0", cfg["fit_config"]["c0"]) # fix=True)
            fitter.set_background_initpar(0, "c1", cfg["fit_config"]["c1"]) # fix=True)

        fit_res = fitter.mass_zfit()
        sgn_sweights = fitter.get_sweights()['signal']

    except:
        fitter = F2MassFitter(data_handler, sgn_func, ["nobkg"], verbosity=0, name=fitter_name)
        fitter.set_signal_initpar(0, "mu", cfg["fit_config"]["mean"])
        fitter.set_signal_initpar(0, "sigma", cfg["fit_config"]["sigma"])
        if sgn_func[0] == "doublecb":
            fitter.set_signal_initpar(0, "alphal", cfg["fit_config"]["alphal"], limits=[0.5, 10])
            fitter.set_signal_initpar(0, "alphar", cfg["fit_config"]["alphar"], limits=[0.5, 10])
            fitter.set_signal_initpar(0, "nl", cfg["fit_config"]["nl"], limits=[1, 30])
            fitter.set_signal_initpar(0, "nr", cfg["fit_config"]["nr"], limits=[1, 30])

        fit_res = fitter.mass_zfit()
        sgn_sweights = None
    
    with open(f"{cfg['output']['dir']}/failed_fits.txt", "a") as f:
        computed_sweights = True if sgn_sweights is not None else False
        f.write(
                f"{fitter_name}: fit_res.valid -> {fit_res.valid}, "
                f"fit_res.status -> {fit_res.status}, "
                f"fit_res.converged -> {fit_res.converged}, "
                f"sweights computed -> {computed_sweights} \n"
               )

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
        show_extra_info = fitter._name_background_pdf_[0] != "nobkg" and fitter.get_background()[1] != 0,
        figsize=(8, 8), extra_info_loc=loc,
        axis_title=ax_title,
        logy=True
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

    return fitter, sgn_sweights

def draw_pid_distributions(dfs, cfg, labels, pt_min, pt_max, sub_dir):
    for var in cfg['variables_to_plot']:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        for df, label in zip(dfs, labels):
            weight = df['w_splot'] if "w_splot" in df.columns else None
            df[var].hist(bins=100, label=label, weights=weight, alpha=0.5, density=True, range=(-5,5))
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
            if "w_splot" in df.columns and sum(df['w_splot'])>0:
                n_sel = np.sum(df.query(f'abs({var}) < {nsigma}')['w_splot'])
                n_total = np.sum(df.query(f'abs({var}) < 5')['w_splot'])
            else:
                n_sel = len(df.query(f'abs({var}) < {nsigma}'))
                n_total = len(df.query(f'abs({var}) < 5'))
            eff = n_sel / n_total if n_total > 0 else 0
            eff_unc = np.sqrt(eff * (1 - eff) / n_total) if n_total > 0 else 0
            effs[-1].append(eff)
            effs_unc[-1].append(eff_unc)

    return effs, effs_unc

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

def run_pt_bin(pt_min, pt_max, cfg, out_daudir, dau_axis_pt, selection, data_df, mc_df, eff_df_sel_row, sel, sel_var):
    dau_pt_sel = selection + f'{pt_min} < {dau_axis_pt} < {pt_max}'
    
    df_data_pt = data_df.query(dau_pt_sel)
    df_mc_pt = mc_df.query(dau_pt_sel)
    
    eff_df_row = [*eff_df_sel_row] + [f"[{pt_min}, {pt_max})"]
    eff_df_mc_row = [*eff_df_sel_row] + [f"[{pt_min}, {pt_max})"]

    fitter, sgn_sweights = fit_mass(df_data_pt, 'data', pt_min, pt_max, sel, cfg, out_daudir, sel_var)

    if sgn_sweights is not None and not np.isclose(fitter.get_background()[0], 0, atol=1):
        df_data_pt = df_data_pt.copy()
        df_data_pt.loc[:, 'w_splot'] = sgn_sweights

        for var in cfg["variables_to_plot"]:
            effs, effs_uncs = get_efficiency([df_data_pt, df_mc_pt], var)
            eff_df_row = eff_df_row + [effs[0]] + [effs_uncs[0]]
            eff_df_mc_row = eff_df_mc_row + [effs[1]] + [effs_uncs[1]]

        if cfg.get('draw_corr'):
            draw_correlation_pt(df_data_pt, 'data', pt_min, pt_max, cfg, out_daudir)
            draw_correlation_pt(df_mc_pt, 'mc', pt_min, pt_max, cfg, out_daudir)

        draw_pid_distributions([df_data_pt, df_mc_pt], cfg, ['data', 'mc'], pt_min, pt_max, out_daudir)
        for var in cfg['variables_to_plot']:
            mean_data, sigma_data = get_distribution_mean_sigma(df_data_pt, var)
            mean_mc, sigma_mc = get_distribution_mean_sigma(df_mc_pt, var)
            eff_df_row = eff_df_row + [mean_data, sigma_data]
            eff_df_mc_row = eff_df_mc_row + [mean_mc, sigma_mc]
    else:
        for var in cfg["variables_to_plot"]:
            eff_df_row = eff_df_row + [[0,0,0]] + [[0,0,0]]
            eff_df_mc_row = eff_df_mc_row + [[0,0,0]] + [[0,0,0]]

        for var in cfg['variables_to_plot']:
            eff_df_row = eff_df_row + [0, 0]
            eff_df_mc_row = eff_df_mc_row + [0, 0] 

    return eff_df_row, eff_df_mc_row

def draw_distributions(cfg_file_name):
    # Read the configuration file
    with open(cfg_file_name, 'r') as cfg_file:
        cfg = yaml.safe_load(cfg_file)

    pt_bins = cfg["pt_bins"]
    selection_vars, ev_sels = get_selections(cfg)

    eff_df_cols = selection_vars + ["fPt"] + [
        item for pair in zip(
            cfg["variables_to_plot"],
            [f"{var}_unc" for var in cfg["variables_to_plot"]]         
            ) for item in pair
    ] + [ item for pair in zip(
            [f"{var}_mean" for var in cfg["variables_to_plot"]],
            [f"{var}_std" for var in cfg["variables_to_plot"]]  
            ) for item in pair
    ]

    eff_dfs = {dau: [] for dau in cfg['dau_names']}
    eff_dfs_mc = {dau: [] for dau in cfg['dau_names']}

    # Read the input data
    data_df = pd.read_parquet(cfg['inputs']['data'])
    mc_df = pd.read_parquet(cfg['inputs']['mc'])

    # Create the file and write the first line
    if not os.path.exists(cfg['output']['dir']):
        os.makedirs(cfg['output']['dir'])
    with open(f"{cfg['output']['dir']}/failed_fits.txt", "w") as file:
        file.write("Failed fit configurations\n")

    for sel in ev_sels:
        selection = f"{cfg['mass_range'][0]} < {cfg['mother_mass_var_name']} < {cfg['mass_range'][1]} and "
        out_dir = ""
        eff_df_sel_row = []
        for var_name, range in zip(selection_vars, sel):
            selection += f'{range[0]} < {var_name} < {range[1]} and '
            # eff_df_sel_row.append(pd.Interval(range[0], range[1], closed='left')) 
            eff_df_sel_row.append(f"[{range[0]}, {range[1]})") 
            out_dir += f"{var_name}_{range[0]}_{range[1]}/" 
        results = []
        workers = cfg['max_workers'] if cfg.get('max_workers') else 2
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for dau, dau_axis_pt in zip(cfg['dau_names'], cfg['dau_pt_var_names']):
                for pt_min, pt_max in zip(pt_bins[:-1], pt_bins[1:]):
                    out_daudir = f"{dau}/" + out_dir
                    print("*******************************")
                    print(f"CONFIG: {eff_df_sel_row}, {pt_min} < pt < {pt_max}")
                    print("*******************************")
                    results.append((executor.submit(run_pt_bin, pt_min, pt_max, cfg, out_daudir, dau_axis_pt, selection, data_df, mc_df, eff_df_sel_row, sel, selection_vars), dau))
                # draw_efficiencies([data_df, mc_df], cfg, ['data', 'mc'], pt_bins, out_daudir)

        for result, dau in results:
            eff, eff_mc = result.result()
            eff_dfs[dau].append(eff)
            eff_dfs_mc[dau].append(eff_mc)

    for dau in cfg['dau_names']:
        eff_df = pd.DataFrame(eff_dfs[dau], columns=eff_df_cols)
        eff_df_mc = pd.DataFrame(eff_dfs_mc[dau], columns=eff_df_cols)
        eff_df.to_parquet(f"{cfg["output"]["dir"]}/{dau}_eff_df.parquet")
        eff_df_mc.to_parquet(f"{cfg["output"]["dir"]}/{dau}_eff_df_mc.parquet")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Draw distributions')
    parser.add_argument('config_file', help='Path to the input configuration file')
    args = parser.parse_args()

    draw_distributions(args.config_file)