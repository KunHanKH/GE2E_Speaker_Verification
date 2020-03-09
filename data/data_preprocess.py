import glob
import os
import numpy as np
import librosa
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from utils.parse_config import config_param
from utils.utils import audio_set_argument, STFT_mel_audio_data


def data_preprocess_tdsv():

    data_config = config_param.data.TD_SV_data
    train_config = config_param.train.TD_SV_train

    print("Start to TD-SV feature extraction.")

    os.makedirs(data_config.train_path_processed, exist_ok=True)
    os.makedirs(data_config.test_path_processed, exist_ok=True)

    audio_set = glob.glob(os.path.dirname(data_config.unprocessed_data))
    total_speaker_num = len(audio_set)
    train_speaker_num = int(len(audio_set)*0.9)
    test_speaker_num = total_speaker_num - train_speaker_num
    print("Total speaker number: {}".format(len(audio_set)))
    print("Train speaker: {}\nTest speaker: {}".format(train_speaker_num, test_speaker_num))

    for i, speaker_dir in enumerate(audio_set):
        print("Processing speaker {}".format(i))

        raw_data_per_speaker = []
        utter_per_speaker = []
        wav_num = 0

        utter_paths = [os.path.join(speaker_dir, utter_name) for utter_name in os.listdir(speaker_dir) if os.path.splitext(utter_name)[1] == ".wav"]
        # extract raw audio data
        for utter_path in utter_paths:
            wav_num += 1
            # print("Processing {}-th speaker's {}-th wav file".format(i, wav_num))
            utter, sr = librosa.core.load(utter_path, data_config.sr)
            raw_data_per_speaker.append(utter)
        # Argument raw data if length is less than default number
        if len(raw_data_per_speaker) < train_config.M:
            raw_data_per_speaker = audio_set_argument(raw_data_per_speaker, train_config.M)

        # Process draw_data after argumentation by STFT and MEL
        for utter in raw_data_per_speaker:
            utter_trim, index = librosa.effects.trim(utter, top_db=30)
            duration = len(utter_trim)/data_config.sr
            utter_strech = librosa.effects.time_stretch(utter_trim, rate=duration/data_config.duration)
            assert len(utter_strech) > 0, "top_db is set to low, make entire utter as silence."

            S = STFT_mel_audio_data(data_config, utter_strech)

            S = S[:data_config.frame, :]
            utter_per_speaker.append(S)

        assert len(utter_per_speaker) > 0, "Can't extract wav feature for speaker {}.".format(i)
        assert len(utter_per_speaker) >= train_config.M, "TD-SV data argument failed for speaker {}".format(i)

        # transform list to np array to save as npy file
        utter_per_speaker = np.array(utter_per_speaker)
        print("Speaker {} has utter set shape: ".format(i), utter_per_speaker.shape)
        if i < train_speaker_num:  # save spectrogram as numpy file
            np.save(os.path.join(data_config.train_path_processed, "speaker%d.npy" % i), utter_per_speaker)
        else:
            np.save(os.path.join(data_config.test_path_processed, "speaker%d.npy" % (i - train_speaker_num)),
                    utter_per_speaker)


def data_preprocess_tisv():

    data_config = config_param.data.TI_SV_data
    train_config = config_param.train.TI_SV_train

    print("Start to TI-SV feature extraction.")

    os.makedirs(data_config.train_path_processed, exist_ok=True)
    os.makedirs(data_config.test_path_processed, exist_ok=True)

    audio_set = glob.glob(os.path.dirname(data_config.unprocessed_data))
    total_speaker_num = len(audio_set)
    train_speaker_num = int(len(audio_set) * 0.9)
    test_speaker_num = total_speaker_num - train_speaker_num
    print("Total speaker number: {}".format(len(audio_set)))
    print("Train speaker: {}\nTest speaker: {}".format(train_speaker_num, test_speaker_num))

    for i, speaker_dir in enumerate(audio_set):

        print("Processing speaker {}".format(i))
        raw_data_per_speaker = []
        utter_per_speaker = []
        wav_num = 0

        utter_paths = [os.path.join(speaker_dir, utter_name) for utter_name in os.listdir(speaker_dir) if os.path.splitext(utter_name)[1] == ".wav"]
        for utter_path in utter_paths:
            wav_num += 1
            utter, sr = librosa.core.load(utter_path, data_config.sr)

            intervals = librosa.effects.split(utter, top_db=30)

            intervals = [interval for interval in intervals if
                         interval[1] - interval[0] > data_config.sr * data_config.duration]

            for interval in intervals:
                raw_data_per_speaker.append(utter[interval[0]:interval[1]])

        assert len(raw_data_per_speaker) > 0, "split top_db too high!"
        if len(raw_data_per_speaker) < train_config.M//2:
            raw_data_per_speaker = audio_set_argument(raw_data_per_speaker, train_config.M//2)

        for utter in raw_data_per_speaker:
            S = STFT_mel_audio_data(data_config, utter)
            utter_per_speaker.append(S[:data_config.frame])
            utter_per_speaker.append(S[-data_config.frame:])

        assert len(utter_per_speaker) > 0, "Can't extract wav feature for speaker {}.".format(i)
        assert len(utter_per_speaker) >= train_config.M, "TD-SV data argument failed for speaker {}".format(i)

        # transform list to np array to save as npy file
        utter_per_speaker = np.array(utter_per_speaker)
        print("Speaker {} has utter set shape: ".format(i), utter_per_speaker.shape)
        if i < train_speaker_num:  # save spectrogram as numpy file
            np.save(os.path.join(data_config.train_path_processed, "speaker%d.npy" % i), utter_per_speaker)
        else:
            np.save(os.path.join(data_config.test_path_processed, "speaker%d.npy" % (i - train_speaker_num)),
                    utter_per_speaker)


data_preprocess_tdsv()
# data_preprocess_tisv()


