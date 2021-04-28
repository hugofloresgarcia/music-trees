""" analyze.py - experiment analysis script"""
import music_trees as mt

import glob
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def bar_with_error(df: pd.DataFrame, dv: str, iv: str, cond: str):
    """
    dv --> dependent variable
    iv --> independent variable
    cond --> conditions (groupings)
    """

    # get all possible values for the IV and conditions
    all_trials = df[iv].unique()
    all_conds = df[cond].unique()

    means = {c: [] for c in all_conds}
    stds = {c: [] for c in all_conds}
    for trial in all_trials:
        for c in all_conds:
            values = df[(df[iv] == trial) & (df[cond] == c)]

            means[c].append(np.mean(values))
            stds[c].append(np.std(values))

    raise NotImplementedError


def analyze(df: pd.DataFrame):
    """
    run a full analysis and model comparison given
    a DataFrame with multiple results
    """
    metrics = df['metric'].unique()
    task_tags = df['tag'].unique()
    n_shots = df['n_shot'].unique()
    models = df['model'].unique()

    for tag in task_tags:
        for metric in metrics:
            # get the df subset with this
            # metric and tag
            subset = df[(df.tag == tag) & (df.metric == metric)]
            breakpoint()

            bar_with_error(df, dv='value', iv='taxonomy_name', conds='n_shot')

            raise NotImplementedError

    raise NotImplementedError


def analyze_folder(path_to_results: str):
    """
    Will look for .csv files recursively and create a single
    DataFrame for all of them
    """

    filepaths = glob.glob(
        str(Path(path_to_results) / '**/*.csv'), recursive=True)
    df = pd.concat([pd.read_csv(fp) for fp in filepaths], ignore_index=True)

    return analyze(df)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('path_to_results', type=str,
                        help='path to the folder containing result csvs')
    args = parser.parse_args()

    analyze_folder(**vars(args))
