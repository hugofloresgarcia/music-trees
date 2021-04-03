import music_trees as mt
from music_trees.utils.data import get_one_hot
from nussl import AudioSignal
import nussl.datasets.transforms as transforms

import argparse
from collections import OrderedDict

import torch
import librosa
import numpy as np

class EpisodicTransform:

    def __init__(self):
        """ 
        wrapper for using regular single-example transforms with an episodic
        dataset

        the provided audio_tfm must convert the AudioSignals to numpy arrays!!!!!
        """ 

    def __call__(self, episode):
        # get the list of classes
        classlist = episode['classes']
        x = np.stack([e['audio'] for e in episode['records']])
        labels = [e['label'] for e in episode['records']]
        # only grab targets for the query set
        proto_target = np.stack([np.argmax(get_one_hot(e['label'], classlist), axis=0) 
                            for e in episode['records'][episode['n_shot']*episode['n_class']:]])
        episode.update(dict(x=x, labels=labels, proto_target=proto_target))
        return episode

class RandomEffects:

    def __init__(self, effect_chain=None):
        self.effect_chain = effect_chain
    
    def __call__(self, signal: AudioSignal):
        _validate_audio_signal(signal)
        return mt.utils.effects.augment_from_audio_signal(signal, self.effect_chain)

class LogMelSpec:

    def __init__(self, hop_length: int, win_length: int):
        self.hop_length = hop_length
        self.win_length = win_length

    def __call__(self, entry: dict):
        signal = entry['audio']
        _validate_audio_signal(signal)
        # downmix
        try:
            x = signal.to_mono().audio_data
        except:
            raise ValueError
        x = x[0]
        assert x.ndim == 1

        spec = librosa.feature.melspectrogram(y=x, sr=signal.sample_rate,
                                              hop_length=self.hop_length, win_length=self.win_length,
                                              power=1)

        spec = np.expand_dims(spec, 0)

        entry['audio'] = spec
        return entry
    
    def __repr__(self):
        return f'logmel-win{self.win_length}-hop{self.hop_length}'

def _validate_audio_signal(signal: AudioSignal):
    assert isinstance(signal, AudioSignal), \
        f'entry["audio"] must be AudioSignal'
    assert signal.has_data
