import numpy as np
import librosa

def audio_arugment(utter):
    '''
    argument audio data by adding noise and stretch
    '''
    noise = np.random.randn(len(utter))
    noise_ration = 0.005
    utter_noise = utter + noise_ration * noise
    stretch_ratio = np.random.randn(1)[0] * 0.05 + 1
    utter_stretch = librosa.effects.time_stretch(utter_noise, rate=stretch_ratio)
    return utter_stretch

def audio_set_argument(raw_utter_set, target_utter_num):
    for i in range(target_utter_num - len(raw_utter_set)):
        index = np.random.randint(0, len(raw_utter_set), 1)[0]
        raw_utter_set.append(audio_arugment(raw_utter_set[index]))
    return raw_utter_set

def STFT_mel_audio_data(data_config, raw_utter):
    S = librosa.stft(raw_utter, n_fft=data_config.nfft, hop_length=int(data_config.hop * data_config.sr),
                     win_length=int(data_config.window * data_config.sr))
    S = np.abs(S) ** 2
    mel_basis = librosa.filters.mel(sr=data_config.sr, n_fft=data_config.nfft, n_mels=data_config.nmels)
    S = np.log10(np.dot(mel_basis, S) + 1e-6).T
    assert S.shape[0] >= data_config.frame, "can't generate 80 frames."
    return S # -> shape [num_frame, data_config.nmel]