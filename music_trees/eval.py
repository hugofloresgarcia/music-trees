import music_trees as mt
from music_trees.tree import MusicTree

import glob

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score


DATASET = 'mdb'
NUM_WORKERS = 0
N_EPISODES = 1
N_CLASS = 12
N_QUERY = 2 * 60  # (2 minutes of audio per class)
N_SHOT = tuple(reversed((1, 2, 4, 8, 16, 32)))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def evaluate(name: str, version: int):
    exp_dir = mt.train.get_exp_dir(name, version)
    assert exp_dir.exists()

    output_dir = exp_dir / 'tests'
    output_dir.mkdir(exist_ok=True)

    ckpt_path = get_ckpt_path(exp_dir)
    ckpt = torch.load(ckpt_path)
    model = load_model_from_ckpt(ckpt_path)
    model = model.to(DEVICE)

    tree = model.model.tree

    # setup transforms
    audio_tfm = mt.preprocess.LogMelSpec(hop_length=mt.HOP_LENGTH,
                                         win_length=mt.WIN_LENGTH)

    all_results = []
    for n_shot in N_SHOT:

        # load our evaluation dataset
        dm = mt.data.MetaDataModule(
            name=DATASET, batch_size=1, num_workers=NUM_WORKERS,
            n_episodes=N_EPISODES, n_class=N_CLASS,
            n_shot=n_shot, n_query=N_QUERY, audio_tfm=audio_tfm,
        )
        dm.setup('test')

        outputs = []
        for index, episode in tqdm(enumerate(dm.test_dataloader())):
            episode = batch2cuda(episode)
            output = model.eval_step(episode, index)
            outputs.append(output)

        results = pd.DataFrame(episode_metrics(outputs, tree))
        results['model'] = f'{name}_v{version}'
        results['n_shot'] = n_shot
        results['n_class'] = N_CLASS
        all_results.append(results)

        all_results_df = pd.concat(all_results)

        results_path = mt.RESULTS_DIR / DATASET / f'{name}-v{version}.csv'
        results_path.parent.mkdir(exist_ok=True, parents=True)

        for key, val in vars(ckpt['hyper_parameters']['args']).items():
            if key in ('model', 'n_shot', 'n_class'):
                continue
            all_results_df[key] = val
        all_results_df.to_csv(results_path)
        print(all_results_df)


def get_ckpt_path(exp_dir):
    ckpts = glob.glob(str(exp_dir / 'checkpoints' / '*.ckpt'))
    assert len(ckpts) == 1
    return ckpts[0]


def load_model_from_ckpt(ckpt_path):
    return mt.models.task.MetaTask.load_from_checkpoint(ckpt_path)


def batch2cuda(batch):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(DEVICE)
        elif isinstance(v, dict):
            batch[k] = batch2cuda(v)
    return batch


def idx2label(labels: torch.Tensor,  classlist: list):
    return [classlist[l] for l in labels]


def episode_metrics(outputs: dict, tree: MusicTree = None):
    """ 
    compute per-episode metrics, and return first and 
    second order statistics for the results
    """

    #  gather a concatenated list of all preds and targets
    task_tags = [t['tag'] for t in outputs[0]['tasks']]
    tasks = {t: [] for t in task_tags}

    results = []
    for index, epi in enumerate(outputs):
        for t in epi['tasks']:
            classlist = t['classlist']
            pred = idx2label(t['pred'], classlist)
            target = idx2label(t['target'], classlist)

            # f1 micro
            results.append({
                'episode_idx': index,
                'metric': 'f1_micro',
                'value': f1_score(target, pred, average='micro', labels=classlist),
                'tag': t['tag'],
            })

            # f1 macro
            results.append({
                'episode_idx': index,
                'metric': 'f1_macro',
                'value': f1_score(target, pred, average='macro', labels=classlist),
                'tag': t['tag'],
            })

            if 'tree' not in t['tag']:
                results.append({
                    'episode_idx': index,
                    'metric': 'hlca-mistake',
                    'value': np.mean([tree.hlca(p, tgt) for p, tgt in zip(pred, target)
                                      if p != tgt]),
                    'tag': t['tag'],
                })

    return pd.DataFrame(results)


def flat_metrics(outputs: dict, tree: MusicTree = None):

    #  gather a concatenated list of all preds and targets
    task_tags = [t['tag'] for t in outputs[0]['tasks']]
    tasks = {t: {'pred': [], 'target': []} for t in task_tags}

    for o in outputs:
        for t in o['tasks']:
            classlist = t['classlist']
            pred = idx2label(t['pred'],  classlist)
            target = idx2label(t['target'], classlist)
            tasks[t['tag']]['pred'].extend(pred)
            tasks[t['tag']]['target'].extend(target)

    metrics = {tag: {} for tag in tasks.keys()}
    for tag, t in tasks.items():
        classlist = list(set(t['target']))
        metrics[tag]['f1_micro'] = f1_score(t['target'], t['pred'],
                                            average='micro', labels=classlist)
        metrics[tag]['f1_macro'] = f1_score(t['target'], t['pred'],
                                            average='macro',  labels=classlist)
        if tree is not None:
            if 'tree' not in tag:
                # only take into account mistakes
                metrics[tag]['hlca-mistake'] = np.mean(
                    [tree.hlca(p, tgt) for p, tgt in zip(t['pred'], t['target'])
                        if p != tgt])

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # add training script arguments
    parser.add_argument('--name', type=str, required=True,
                        help='name of the experiment')
    parser.add_argument('--version', type=int, required=True,
                        help='version.')

    args = parser.parse_args()
    evaluate(**vars(args))
