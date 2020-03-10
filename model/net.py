import torch
import torch.nn as nn
import os
import sys
if os.path.join(os.path.dirname(__file__),'..') not in sys.path:
    sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from utils.parse_config import config_param

class SpeechEmbedder(nn.Module):

    def __init__(self):
        super(SpeechEmbedder, self).__init__()

        if not config_param.key_word:
            self.data_config = config_param.data.TI_SV_data
            self.model_config = config_param.model.TI_SV_model
        else:
            self.data_config = config_param.data.TD_SV_data
            self.model_config = config_param.model.TD_SV_model

        self.LSTM = nn.LSTM(input_size=self.data_config.nmels, hidden_size=self.model_config.hidden, num_layers=self.model_config.num_layer, batch_first=True)
        self.projection = nn.Linear(in_features=self.model_config.hidden, out_features=self.model_config.proj)

    def forward(self, utter):

        x, _ = self.LSTM(utter)
        x = x[:, -1]
        x = self.projection(x)
        x = x / torch.norm(x, p=2, dim=1).unsqueeze(dim=1) # L2 normalization
        return x


# test_input = torch.randn((100, 50, 40))
# net = SpeechEmbedder()
# embedder = net(test_input)
# print(embedder.shape)
