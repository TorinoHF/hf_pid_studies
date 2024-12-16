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

def skim_tree(treefile, treename, selection):
    extension = os.path.splitext(treefile)[1]
    if extension == ".root":
        # Load the tree
        with uproot.open(treefile, encoding="utf8") as f:
            df = []
            for folder_name, folder in f.items():
                if treename in folder_name:
                    df.append(folder.arrays(library="pd"))
            df = pd.concat(df)
        print("Loaded data from .root file!")
    else:
        df = pd.read_parquet(treefile)
        print("Loaded data from .parquet file!")

    if 'fCascCosPA' in df.columns:
        selection = selection.replace("fCascCosPa", "fCascCosPA")
    if 'fDCAV0daughters' in df.columns:
        selection = selection.replace("fDcaV0Daughters", "fDCAV0daughters")
    if 'fDCAv0topv' in df.columns:
        selection = selection.replace("fDcaV0ToPv", "fDCAv0topv")
    if selection != "":
        df = df.query(selection)
    
    return df

# Load configuration file
with open(args.cfg, "r") as stream:
    selections_cfg = yaml.safe_load(stream)

for itree in selections_cfg['trees']:

    treename = itree['treename'] if itree.get('treename') else ""
    suffix = f"_{itree['suffix']}" if itree.get('suffix') else ""
    
    print("---------------------------------")
    print(f"Loading data from {itree['treefiledata']}...")
    print(f"Selection: {itree['selections'] if itree.get('selections') else 'None'}")
    selection_data = f"{itree['selections']}" if itree.get('selections') else "" 
    df_data = skim_tree(itree['treefiledata'], treename, selection_data)

    df_data_basename = os.path.splitext(os.path.basename(itree["treefiledata"]))[0]
    if not os.path.exists(f"{itree['outdir']}/data/"):
        os.makedirs(f"{itree['outdir']}/data/")
    
    df_data.to_parquet(f"{itree['outdir']}/data/{df_data_basename}{suffix}.parquet")
    print(f"Saved skimmed tree to: {f"{itree['outdir']}/data/{df_data_basename}{suffix}.parquet"}")

    if itree.get('treefilemc'):
        print(f"Loading data from {itree['treefilemc']}...")
        selection_mc = selection_data + f" and fCandFlag == {itree['candflag']}" if itree.get('selections') else f"fCandFlag == {itree['candflag']}"
        df_mc = skim_tree(itree['treefilemc'], treename, selection_mc)
        
        df_mc_basename = os.path.splitext(os.path.basename(itree["treefilemc"]))[0]
        if not os.path.exists(f"{itree['outdir']}/mc/"):
            os.makedirs(f"{itree['outdir']}/mc/")

        df_mc.to_parquet(f"{itree['outdir']}/mc/{df_mc_basename}{suffix}.parquet")
        print(f"Saved skimmed tree to: {f"{itree['outdir']}/mc/{df_mc_basename}{suffix}.parquet"}")
