"""
Project a tree from a root file to a parquet file.

Run:
    python project_tree.py config_file
"""

import argparse
import os

import uproot
import pandas as pd
import yaml


def enforce_list(var):
    """
    Ensure that the input variable is returned as a list.

    Args:
        var: The input variable to be enforced as a list.

    Returns:
        list: A list containing the input variable if it was not already a list, or the input
        variable itself if it was already a list.
    """
    if not isinstance(var, list):
        return [var]
    return var


def get_cuts(cuts_config):
    """
    Generate a string of cuts based on the provided configuration.

    Args:
        cuts_config (dict): The cuts configuration

    Returns:
        str: A string representing the cuts
    """
    # Extract variable names and cuts from the config
    var_names = list(cuts_config.keys())

    cuts_list = [cuts_config[var] for var in var_names]
    names_list = [cut['name'] for cut in cuts_list]
    mins_list = [[range['min'] for range in cut['ranges']] for cut in cuts_list]
    maxs_list = [[range['max'] for range in cut['ranges']] for cut in cuts_list]

    mins_list = enforce_list(mins_list)
    maxs_list = enforce_list(maxs_list)

    cuts = ""
    for mins_var, maxs_var, name in zip(mins_list, maxs_list, names_list):
        cuts += "("
        for min_var, max_var in zip(mins_var, maxs_var):
            cuts += f"{min_var} < {name} < {max_var} or "
        cuts = cuts[:-3] + ") and "

    return cuts[:-5]


def project_tree(config_file):
    """
    Load and filter data and MC trees, then save the resulting dataframes to .parquet files.
    Args:
        config_file (str): Path to the YAML configuration file.
    """
    with open(config_file, 'r', encoding="utf8") as f:
        config = yaml.safe_load(f)

    with open(config["cutset"], 'r', encoding="utf8") as cuts_config_file:
        cuts_config = yaml.load(cuts_config_file, yaml.FullLoader)

    cuts = get_cuts(cuts_config)

    # Load the tree
    print(f"Loading data from {config['inputs']['data']}...")
    with uproot.open(config["inputs"]["data"], encoding="utf8") as f:
        df_data = []
        for folder_name, folder in f.items():
            if config["inputs"]["tree_name"] in folder_name:
                df_data.append(folder.arrays(library="pd"))
        df_data = pd.concat(df_data)
    df_data = df_data.query(cuts)
    print("Loaded data.")

    print("Saving data to parquet...")
    outpath = os.path.join(
        config["outputs"]['dir'],
        f"skimmed_{os.path.basename(config['inputs']['data']).split('_')[1]}_{config['outputs']['suffix']}.parquet"  # pylint: disable=line-too-long, # noqa: E501
    )
    df_data.to_parquet(outpath)
    del df_data
    print("Data saved.")

    print(f"Loading MC from {config['inputs']['mc']}...")
    with uproot.open(config["inputs"]["mc"], encoding="utf8") as f:
        df_mc = []
        for folder_name, folder in f.items():
            if config["inputs"]["tree_name"] in folder_name:
                df_mc.append(folder.arrays(library="pd"))
        df_mc = pd.concat(df_mc)
    df_mc = df_mc.query(cuts)
    flag_sel = ""
    for value in config["flag"]["values"]:
        flag_sel += f"abs({config['flag']['name']}) == {value} or "
    df_mc = df_mc.query(flag_sel[:-3])
    print("Loaded MC.")

    print("Saving MC to parquet...")
    outpath = os.path.join(
        config["outputs"]['dir'],
        f"skimmed_{os.path.basename(config['inputs']['mc']).split('_')[1]}_{config['outputs']['suffix']}.parquet"  # pylint: disable=line-too-long, # noqa: E501
    )
    df_mc.to_parquet(outpath)
    del df_mc
    print("MC saved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Project a tree from a root file')
    parser.add_argument('config_file', type=str, help='Path to the config file')
    args = parser.parse_args()

    project_tree(args.config_file)
