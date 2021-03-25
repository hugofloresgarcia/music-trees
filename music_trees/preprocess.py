import music_trees as mt
from music_trees.utils.data import get_one_hot
from nussl import AudioSignal
import nussl.datasets.transforms as transforms

import argparse

import torch
import librosa
import numpy as np

transforms.Cache

class EpisodicTransform:

    def __init__(self, audio_tfm, n_class: int, n_shot: int, n_query: int):
        """ 
        wrapper for using regular single-example transforms with an episodic
        dataset

        the provided audio_tfm must convert the AudioSignals to numpy arrays!!!!!
        """ 
        self.n_class = n_class
        self.n_query = n_query
        self.n_shot = n_shot

        self.audio_tfm = audio_tfm
    
    def __call__(self, episode):
        # get the list of classes
        classlist = episode['classes']

        # process support set
        support = []
        for name, examples in episode['support'].items():
            class_support = np.stack([self.audio_tfm(e['audio']) for e in examples])
            support.append(class_support)
        episode['support'] = np.stack(support)


        # process query set
        episode['target'] = np.stack([np.argmax(get_one_hot(e['label'], classlist), axis=0) for e in episode['query']])
        episode['query'] = np.stack([self.audio_tfm(e['audio']) for e in episode['query']])

        return episode

class RandomEffects:

    def __init__(self, effect_chain=None):
        self.effect_chain = effect_chain
    
    def __call__(self, signal: AudioSignal):
        return mt.utils.augment_from_audio_signal(signal, self.effect_chain)

class LogMelSpec:

    def __init__(self, hop_length: int, win_length: int):
        self.hop_length = hop_length
        self.win_length = win_length

    def __call__(self, signal: AudioSignal):
        # downmix
        try:
            x = signal.to_mono().audio_data
        except:
            breakpoint()
        x = x[0]
        assert x.ndim == 1

        spec = librosa.feature.melspectrogram(y=x, sr=signal.sample_rate,
                                              hop_length=self.hop_length, win_length=self.win_length,
                                              power=1)
        
        # add a channel dimension
        spec = np.expand_dims(spec, 0)

        return spec
