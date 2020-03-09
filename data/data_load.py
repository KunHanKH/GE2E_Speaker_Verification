import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import sys
import os
if os.path.join(os.path.dirname(__file__),'..') not in sys.path:
    sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from utils.parse_config import config_param

class SpeechDataset(Dataset):
    def __init__(self):

        if not config_param.key_word:
            self.data_config = config_param.data.TI_SV_data
            self.train_config = config_param.train.TI_SV_train
            self.frame_length = self.data_config.frame

        else:
            self.data_config = config_param.data.TD_SV_data
            self.train_config = config_param.train.TD_SV_train

        if config_param.training:
            self.path = self.data_config.train_path_processed
        else:
            self.path = self.data_config.test_path_processed

        self.speaker_paths = [os.path.join(self.path, speaker) for speaker in os.listdir(self.path)]

    def __len__(self):
        return len(self.speaker_paths)

    def __getitem__(self, idx):

        if not config_param.key_word:
            if idx % self.train_config.N == 0:
                self.frame_length = np.random.randint(self.data_config.frame_low, self.data_config.frame, 1)[0]
        else:
            self.frame_length = self.data_config.frame

        selected_speaker_path = random.sample(self.speaker_paths, 1)[0]

        utter_per_speaker = np.load(selected_speaker_path)
        shuffle_index = np.random.randint(0, utter_per_speaker.shape[0], self.train_config.M)
        utter_per_speaker = utter_per_speaker[shuffle_index]
        utter_per_speaker = utter_per_speaker[:, :self.frame_length]
        utter_per_speaker = torch.tensor(utter_per_speaker)
        return utter_per_speaker

# dataset = SpeechDataset()
# dataloader = DataLoader(dataset, batch_size=dataset.train_config.N)
# for i, batch in enumerate(dataloader):
#     print(batch.shape)

