""" analyze.py - experiment analysis script"""
import music_trees as mt

from collections import OrderedDict
import glob
from pathlib import Path
import random
from itertools import combinations, permutations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
from scipy.stats import wilcoxon

BASELINE_NAME = 'baseline'
ANALYSES_DIR = mt.ROOT_DIR / 'analyses'

ALL_COLORS = ["#ff595e", "#ffca3a", "#8ac926", "#1982c4", "#eaf6ff",
              "#6a4c93", "#ed6a5a", "#f4f1bb", "#9bc1bc", "#5d576b",
              "#e6ebe0", "#ffa400", "#009ffd", "#2a2a72", "#232528", ]
random.shuffle(ALL_COLORS)


def significance(df: pd.DataFrame, dv: str, iv: str, cond: str):
    """ 
    returns a DataFrame with p values for each condition between independent variables
    """
    # get all possible values for the IV and conditions
    all_trials = df[iv].unique()
    all_conds = list(natsorted(df[cond].unique()))
    pairs = list(permutations(all_trials, 2))

    pvals = []
    for co in all_conds:
        for pair in pairs:
            subset = df[df[cond] == co]
            s1, s2 = pair

            df1 = subset[subset[iv] == s1].copy()
            df2 = subset[subset[iv] == s2].copy()

            stat, p = wilcoxon(df1['value'].values, df2['value'].values)
            pvals.append({
                'a': s1,
                'b': s2,
                cond: co,
                'p': p,
                'stat': stat,
                'significant?': p < 0.01
            })

    return pd.DataFrame(pvals)


def bar_with_error(df: pd.DataFrame, dv: str, iv: str,
                   cond: str, title: str = None,
                   xlabel: str = None, ylabel: str = None) -> plt.figure:
    """
    dv --> dependent variable
    iv --> independent variable
    cond --> conditions (groupings)
    """
    plt.rcParams["figure.figsize"] = (7, 4)

    # get all possible values for the IV and conditions
    all_trials = list(natsorted(df[iv].unique()))
    all_conds = list(natsorted(df[cond].unique()))

    bar_width = 0.25 * 3 / len(all_trials)

    means = OrderedDict((tr, []) for tr in all_trials)
    stds = OrderedDict((tr, []) for tr in all_trials)
    for trial in all_trials:
        for c in all_conds:
            # get the list of all scores per episode
            values = df[(df[iv] == trial) & (df[cond] == c)][dv].values

            means[trial].append(np.mean(values))
            stds[trial].append(np.std(values))

    # make a bar plot for each condition
    bp = np.arange(len(list(means.values())[0]))
    bar_pos = OrderedDict((tr, bp + i*bar_width)
                          for i, tr in enumerate(all_trials))
    for idx, tr in enumerate(all_trials):
        plt.bar(bar_pos[tr], means[tr], yerr=stds[tr], width=bar_width, capsize=12*bar_width,
                color=ALL_COLORS[idx], edgecolor='white', label=tr)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(ticks=[i + bar_width for i in range(len(all_conds))],
               labels=all_conds)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    fig = plt.gcf()
    plt.close()

    return fig


def boxplot(df: pd.DataFrame, dv: str, iv: str,
            cond: str, title: str = None,
            xlabel: str = None, ylabel: str = None):

    import seaborn as sns

    sns.boxplot(data=df, x=cond, y=dv, hue=iv, palette="pastel")
    # sns.despine(offset=10, trim=True)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.tight_layout()
    fig = plt.gcf()
    plt.close()
    return fig


def table(df: pd.DataFrame, dv: str, iv: str,
          cond: str, title: str = None,
          xlabel: str = None, ylabel: str = None):

    # get all possible values for the IV and conditions
    all_trials = list(natsorted(df[iv].unique()))
    all_conds = list(natsorted(df[cond].unique()))

    bar_width = 0.25 * 3 / len(all_trials)

    means = OrderedDict((tr, []) for tr in all_trials)
    stds = OrderedDict((tr, []) for tr in all_trials)
    mnstd = OrderedDict((tr, []) for tr in all_trials)
    for trial in all_trials:
        for c in all_conds:
            # get the list of all scores per episode
            values = df[(df[iv] == trial) & (df[cond] == c)][dv].values

            means[trial].append(np.mean(values))
            stds[trial].append(np.std(values))
            mnstd[trial].append(f'{np.mean(values):.4f}±{np.std(values):.4f}')

    tbl = pd.DataFrame()
    tbl['trial'] = all_trials

    for i, c in enumerate(all_conds):
        tbl[c] = [v[i] for v in mnstd.values()]

    return tbl


def epi_below_base(df: pd.DataFrame, tag: str, metric: str):
    # get all possible trials and n-shots
    all_trials = df['name'].unique()
    shots = np.sort(df['n_shot'].unique())

    # track the episodes where the baseline does better than the compared model
    tracked_epis = []

    for n_shot in shots:
        # build baseline subset
        baseline = df[(df.n_shot == n_shot) & (df.name == BASELINE_NAME)]

        for trial in all_trials:
            # model to be compared
            compared_model = df[(df.n_shot == n_shot) & (df.name == trial)]
            for episode in compared_model.episode_idx.unique():
                # compared model's value for the given metric
                compared_val = compared_model[compared_model.episode_idx ==
                                              episode].value.values[0]
                # base model value for the given metric
                based_val = baseline[baseline.episode_idx ==
                                     episode].value.values[0]
                if based_val > compared_val:
                    tracked_epis.append({
                                        'episode': episode,
                                        'tag': tag,
                                        'model-name': compared_model.name.values[0],
                                        'baseline': BASELINE_NAME,
                                        'n-shot': n_shot,
                                        'metric': metric,
                                        'based-val': based_val,
                                        'model-val': compared_val,
                                        'difference': based_val - compared_val
                                        })

    return pd.DataFrame(tracked_epis)


def epi_above_base(df: pd.DataFrame, tag: str, metric: str, take_top: int = 3):
    # get all possible trials and n-shots
    all_trials = df['name'].unique()
    shots = np.sort(df['n_shot'].unique())

    # track the episodes where the compared model does better than the baseline
    tracked_epis = []

    for n_shot in shots:
        # build baseline subset
        baseline = df[(df.n_shot == n_shot) & (df.name == BASELINE_NAME)]

        for trial in all_trials:
            # model to be compared
            compared_model = df[(df.n_shot == n_shot) & (df.name == trial)]
            temp_tracked_epis = []
            for episode in compared_model.episode_idx.unique():
                # compared model's value for the given metric
                compared_val = compared_model[compared_model.episode_idx ==
                                              episode].value.values[0]
                # base model value for the given metric
                based_val = baseline[baseline.episode_idx ==
                                     episode].value.values[0]

                if based_val < compared_val:
                    temp_tracked_epis.append({
                        'episode': episode,
                        'tag': tag,
                        'model-name': compared_model.name.values[0],
                        'baseline': BASELINE_NAME,
                        'n-shot': n_shot,
                        'metric': metric,
                        'based-val': based_val,
                        'model-val': compared_val,
                        'difference': compared_val - based_val
                    })
            # sorting this temp list by the compared models values in descending order
            temp_tracked_epis = sorted(
                temp_tracked_epis, key=lambda tracked: tracked['difference'], reverse=True)

            # using enumerate here instead of slicing for compact error checking
            for i, tracked_epi in enumerate(temp_tracked_epis):
                if i < take_top:
                    tracked_epis.append(tracked_epi)
                else:
                    break

    return pd.DataFrame(tracked_epis)


def barplot_annotate_brackets(num1, num2, data, center, height,
                              yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None):
    """ 
    https://stackoverflow.com/questions/11517986/indicating-the-statistically-significant-difference-in-bar-graph
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    plt.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    plt.text(*mid, text, **kwargs)


def analyze(df: pd.DataFrame, name: str):
    """
    run a full analysis and model comparison given
    a DataFrame with multiple results
    """
    output_dir = ANALYSES_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = df['metric'].unique()
    task_tags = df['tag'].unique()

    for tag in task_tags:
        subdir = output_dir / tag
        subdir.mkdir(exist_ok=True)

        for metric in metrics:
            # get the df subset with this
            # metric and tag
            subset = df[(df.tag == tag) & (df.metric == metric)]
            if subset.empty:
                continue

            dv = 'value'
            iv = 'name'
            cond = 'n_shot'

            tbl = table(subset, dv=dv, iv=iv,
                        cond=cond, title=name,
                        xlabel='n_shot', ylabel=metric)
            tbldir = subdir / 'tables'
            tbldir.mkdir(exist_ok=True)
            tbl.to_markdown(tbldir / f'{metric}.md')

            errorbar = bar_with_error(subset, dv=dv, iv=iv,
                                      cond=cond, title=name,
                                      xlabel='number of support examples', ylabel=metric)
            errordir = subdir / 'error-bars'
            errordir.mkdir(exist_ok=True)
            errorbar.savefig(errordir / f'{metric}.png')

            box = boxplot(subset, dv=dv, iv=iv,
                          cond=cond, title=name,
                          xlabel='number of support examples', ylabel=metric)
            boxdir = subdir / 'boxplots'
            boxdir.mkdir(exist_ok=True)
            box.savefig(boxdir / f'{metric}.png')

            sig_df = significance(subset, dv=dv, iv=iv, cond=cond)
            sig_dir = subdir / 'significance'
            sig_dir.mkdir(exist_ok=True)
            sig_df.to_csv(sig_dir / f'significance-{metric}.csv')

            comp_dir = subdir / 'episode-comparisons'
            comp_dir.mkdir(exist_ok=True)
            epi_above_df = epi_above_base(subset, tag, metric)
            epi_above_df.to_csv(comp_dir / f'episodes-above-base-{metric}.csv')

            epi_below_df = epi_below_base(subset, tag, metric)
            epi_below_df.to_csv(comp_dir / f'episodes-below-base-{metric}.csv')

    return


def analyze_folder(path_to_results: str, name: str):
    """
    Will look for .csv files recursively and create a single
    DataFrame for all of them
    """

    filepaths = glob.glob(
        str(Path(path_to_results) / '**/*.csv'), recursive=True)
    df = pd.concat([pd.read_csv(fp) for fp in filepaths], ignore_index=True)
    df = df.drop_duplicates()
    return analyze(df, name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('path_to_results', type=str,
                        help='path to the folder containing result csvs')
    parser.add_argument('name', type=str,
                        help='name of folder with analysis output. will be under /analyses/')
    args = parser.parse_args()

    analyze_folder(**vars(args))
