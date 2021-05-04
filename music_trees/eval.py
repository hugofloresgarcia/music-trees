from logging import log
from typing import OrderedDict

import music_trees as mt
from music_trees.tree import MusicTree
from music_trees.models.task import plot_confusion_matrix

import glob
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from embviz.logger import EmbeddingSpaceLogger

DATASET = 'mdb'
NUM_WORKERS = 0
N_EPISODES = 100
N_CLASS = 12
N_QUERY = 2 * 60  # (2 minutes of audio per class)
N_SHOT = tuple(reversed((1, 4, 8, 16)))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def evaluate(exp_dir):
    exp_dir = Path(exp_dir)
    assert exp_dir.exists()
    # load the checkpoint
    ckpt_path = get_ckpt_path(exp_dir)
    ckpt = torch.load(ckpt_path)

    # set run's name
    name = mt.generate.core.clean(ckpt['hyper_parameters']['args'].name)

    # define an output dir
    results_dir = results_path = mt.RESULTS_DIR / \
        DATASET / f'{name}'
    results_dir.mkdir(parents=True, exist_ok=True)

    # create embedding space loggers for each test condition
    embloggers = OrderedDict(
        (n, EmbeddingSpaceLogger(str(results_dir / f'space-n_shot={n}'))) for n in N_SHOT)

    # load the model from checkpoint and move to tree
    model = load_model_from_ckpt(ckpt_path)
    model = model.to(DEVICE)

    # get the model's tree
    tree = model.model.tree

    # setup transforms
    audio_tfm = mt.preprocess.LogMelSpec(hop_length=mt.HOP_LENGTH,
                                         win_length=mt.WIN_LENGTH)

    all_results = []
    for n_shot in N_SHOT:

        # load our evaluation dataset
        tt_kwargs = dict(n_episodes=N_EPISODES, n_class=N_CLASS,
                         n_shot=n_shot, n_query=N_QUERY, audio_tfm=audio_tfm,)
        dm = mt.data.MetaDataModule(name=DATASET, batch_size=1,
                                    num_workers=NUM_WORKERS, tt_kwargs=tt_kwargs)
        dm.setup('test')

        epi_classes = []
        outputs = []
        for index, episode in tqdm(enumerate(dm.test_dataloader())):
            # episode should classlist, etc.
            episode = batch2cuda(episode)
            # track the class list for each episode
            epi_classes.append(episode['classlist'])

            # output should have all predictions and targets, etc.
            output = model.eval_step(episode, index)
            outputs.append(output)

            # log the episode on the embedding space
            log_episode_space(embloggers[n_shot], episode, output, index)

        results = pd.DataFrame(episode_metrics(
            outputs,  name=name, results_dir=results_dir, tree=tree, n_shot=n_shot))
        results['model'] = f'{name}'
        results['n_shot'] = n_shot
        results['n_class'] = N_CLASS
        all_results.append(results)

        all_results_df = pd.concat(all_results)

        results_path = results_dir / f'{name}.csv'

        for key, val in vars(ckpt['hyper_parameters']['args']).items():
            if key in ('model', 'n_shot', 'n_class'):
                continue
            all_results_df[key] = val
        all_results_df.to_csv(results_path)
        print(all_results_df)


def log_episode_space(logger: EmbeddingSpaceLogger, episode: dict, output: dict,
                      episode_idx):
    task = [t for t in output['tasks'] if 'tree' not in t['tag']][0]
    tree_tasks = {t['tag']: t for t in output['tasks'] if 'tree' in t['tag']}

    records = output['records']
    n_support = output['n_shot'] * output['n_class']
    n_query = output['n_query'] * output['n_class']

    # class index to class name
    def gc(i): return output['classlist'][i]

    embeddings = [e for e in task['support_embedding']] + \
        [e for e in task['query_embedding']]

    support_labels = [r['label'] for r in output['records']][:n_support]
    preds = support_labels + [gc(i) for i in task['pred']]
    labels = support_labels + [gc(i) for i in task['target']]

    metatypes = ['support'] * n_support
    for p, tr in zip(preds[n_support:], labels[n_support:]):
        metatype = 'query-correct' if p == tr else 'query-incorrect'
        metatypes.append(metatype)

    audio_paths = [r['audio_path'] for r in records]

    # also visualize the prototypes, though without audio
    for p_emb, label in zip(task['prototype_embedding'], task['classlist']):
        embeddings.append(p_emb)
        labels.append(label)
        preds.append(label)
        metatypes.append('proto')
        audio_paths.append(None)

    embeddings = torch.stack(embeddings).detach().cpu().numpy()

    metadata = dict(audio_path=audio_paths, pred=preds,
                    target=labels, label=labels)

    logger.add_step(step_key=f'{episode_idx}', embeddings=embeddings, symbols=metatypes,
                    metadata=metadata, title='meta space', labels=labels)


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


def episode_metrics(outputs: dict, name: str, results_dir,
                    tree: MusicTree = None, n_shot=None):
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
            tag = t['tag']
            # f1 micro
            results.append({
                'episode_idx': index,
                'metric': 'f1_micro',
                'value': f1_score(target, pred, average='micro', labels=classlist),
                'tag': tag,
            })

            # f1 macro
            results.append({
                'episode_idx': index,
                'metric': 'f1_macro',
                'value': f1_score(target, pred, average='macro', labels=classlist),
                'tag': tag,
            })

            # raw episode accuracy
            results.append({
                'episode_idx': index,
                'metric': 'epi-accuracy',
                'value': accuracy_score(target, pred, normalize=True),
                'tag': tag,
            })

            # creating the confusion matrix for this episode
            conf_matrix = confusion_matrix(
                target, pred, normalize='true')
            fig = plot_confusion_matrix(
                conf_matrix, classlist, title=f'Episode {index} task: {tag}')

            conf_matrix_path = results_dir / \
                'conf_matrices' / \
                f'n_shot={n_shot}' / f'{tag}' / f'{index}'
            conf_matrix_path.parent.mkdir(exist_ok=True, parents=True)

            fig.savefig(conf_matrix_path)

            if 'tree' not in tag:
                # track the highest least common ancestor
                results.append({
                    'episode_idx': index,
                    'metric': 'hlca-mistake',
                    'value': np.mean([tree.hlca(p, tgt) for p, tgt in zip(pred, target)
                                      if p != tgt]),
                    'tag': tag,
                })

                # making variables for hierachical precision and recall
                hP = np.mean([tree.hierarchical_precision(p, tgt)
                              for p, tgt in zip(pred, target)])
                hR = np.mean([tree.hierarchical_recall(p, tgt)
                              for p, tgt in zip(pred, target)])

                # tracking the hierarchical precision
                results.append({
                    'episode_idx': index,
                    'metric': 'hierarchical-precision',
                    'value': hP,
                    'tag': tag,
                })

                # tracking the hierarchical recall
                results.append({
                    'episode_idx': index,
                    'metric': 'hierarchical-recall',
                    'value': hR,
                    'tag': tag,
                })

                # tracking the hierachical f1
                results.append({
                    'episode_idx': index,
                    'metric': 'hierarchical-f1',
                    'value': (2 * hP * hR)/(hP + hR),
                    'tag': tag,
                })

    return pd.DataFrame(results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # add training script arguments
    parser.add_argument('--exp_dir', type=str, required=True,
                        help='experiment directory')

    args = parser.parse_args()
    evaluate(**vars(args))
