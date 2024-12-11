import argparse
import pandas as pd
import numpy as np
import yaml
import uproot
import os
from ROOT import TFile, TH1D

parser = argparse.ArgumentParser(description='Arguments')
parser.add_argument('cfg')
args = parser.parse_args()


# Load configuration file
with open(args.cfg, "r") as stream:
    selections_cfg = yaml.safe_load(stream)
    
if not os.path.exists(os.path.dirname(selections_cfg['histofile'])):
    os.makedirs(os.path.dirname(selections_cfg['histofile']))
outfile = TFile(f"{selections_cfg['histofile']}", "RECREATE")

for itree in selections_cfg['trees']:

    df = pd.read_parquet(itree["treefile"])

    h_mass_before = TH1D('hMass_before', 'hMass_before', itree['histocfg'][0], itree['histocfg'][1], itree['histocfg'][2])
    fill_weigths_before = np.ones(len(df))
    h_mass_before.FillN(len(df)-1, np.asarray(df[f"{itree['masscol']}"], 'd' ), fill_weigths_before)
    h_mass_before.Write()

    sel_df = df.query(f"{itree['selections']}") if itree.get('selections') else df

    h_mass_after = TH1D('hMass_after', 'hMass_after', itree['histocfg'][0], itree['histocfg'][1], itree['histocfg'][2])
    fill_weigths_after = np.ones(len(sel_df))
    h_mass_after.FillN(len(sel_df)-1, np.asarray(sel_df[f"{itree['masscol']}"], 'd' ), fill_weigths_after)
    h_mass_after.Write()

    if not os.path.exists(os.path.dirname(itree['outfile'])):
        os.makedirs(os.path.dirname(itree['outfile']))
    sel_df.to_parquet(f"{itree['outfile']}")

outfile.Close()