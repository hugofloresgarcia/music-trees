""" extensions to nussl.AudioSignal """
from nussl import AudioSignal

import warnings
import copy

import numpy as np
import sox


def trim_silence(signal: AudioSignal, sample_rate: int,
                 min_silence_duration=0.3):
    """ trim silence from audio array using sox
    """
    audio = signal.to_mono(overwrite=False, keep_dims=True).audio_data[0]
    assert audio.ndim == 1
    tfm = sox.Transformer()
    tfm.silence(min_silence_duration=min_silence_duration,
                buffer_around_silence=False)
    audio = tfm.build_array(input_array=np.expand_dims(audio, axis=1),
                            sample_rate_in=sample_rate)
    signal.audio_data = np.expand_dims(audio, axis=0)
    return signal

def _check_audio_types(audio: np.ndarray):
    assert isinstance(
        audio, np.ndarray), f'expected np.ndarray but got {type(audio)} as input.'
    assert audio.ndim == 2, f'audio must be shape (channels, time), got shape {audio.shape}'
    if audio.shape[-1] < audio.shape[-2]:
        warnings.warn(f'got audio shape {audio.shape}. Audio should be (channels, time). \
                        typically, the number of samples is much larger than the number of channels. ')

def window(signal: AudioSignal, window_len: int = 48000, hop_len: int = 4800):
    """split audio into overlapping windows

    note: this is not a memory efficient view like librosa.util.frame. 
    It will return a list of AudioSignals

    Args:
        audio (np.ndarray): audio array with shape (channels, samples)
        window_len (int, optional): [description]. Defaults to 48000.
        hop_len (int, optional): [description]. Defaults to 4800.
    Returns:
        audio_windowed List[AudioSignal]: list of individual windows as AudioSignals
    """
    audio = signal.audio_data
    _check_audio_types(audio)
    # determine how many window_len windows we can get out of the audio array
    # use ceil because we can zero pad
    n_chunks = audio.shape[-1] // window_len
    start_idxs = np.arange(0, n_chunks * window_len, hop_len)

    windows = []
    for start_idx in start_idxs:
        # end index should be start index + window length
        end_idx = start_idx + window_len
        # BUT, if we have reached the end of the audio, stop there
        end_idx = min([end_idx, audio.shape[-1]])
        # create audio window
        win = np.array(audio[:, start_idx:end_idx])
        # zero pad window if needed
        win = zero_pad(win, required_len=window_len)
        windows.append(win)

    empty_sentinel = copy.deepcopy(signal)
    empty_sentinel.audio_data = np.zeros((1, 2000))
    
    signals = []
    for window in windows:
        signal = copy.deepcopy(empty_sentinel)
        signal.audio_data = window
        assert signal.has_data, "internal error during windowing"
        signals.append(signal)

    return signals

def zero_pad(audio: np.ndarray, required_len: int = 48000) -> np.ndarray:
    """zero pad audio array to meet a multiple of required_len
    all padding is done at the end of the array (no centering)

    Args:
        audio (np.ndarray): audio array w shape (channels, sample)
        required_len (int, optional): target length in samples. Defaults to 48000.

    Returns:
        np.ndarray: zero padded audio
    """
    _check_audio_types(audio)

    num_frames = audio.shape[-1]

    before = 0
    after = required_len - num_frames % required_len
    if after == required_len:
        return audio
    audio = np.pad(audio, pad_width=((0, 0), (before, after)),
                   mode='constant', constant_values=0)
    return audio
