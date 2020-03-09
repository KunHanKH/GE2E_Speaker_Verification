import torch
import numpy as np
import torch.nn as nn
import os
import sys
if os.path.join(os.path.dirname(__file__),'..') not in sys.path:
    sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from utils.parse_config import config_param
from utils.utils import get_similarity, get_contrast_loss, get_softmax_loss

class GE2ELoss(nn.Module):

    def __init__(self):
        super(GE2ELoss, self).__init__()

        if not config_param.key_word:
            self.model_config = config_param.model.TI_SV_model
            self.train_config = config_param.train.TI_SV_train
        else:
            self.model_config = config_param.model.TD_SV_model
            self.train_config = config_param.train.TD_SV_train

        self.w = nn.Parameter(torch.tensor(10.0), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(-5.0), requires_grad=True)

    def forward(self, embedding):
        '''
        :param embedding: shape -> [NxM, feature]
        :return:
        '''
        embedding = torch.reshape(embedding, (self.train_config.N, self.train_config.M, self.model_config.proj))
        similarity = self.w * get_similarity(embedding) + self.b # shape -> (N, M, N)

        if self.model_config.loss == "contrast":
            loss = get_contrast_loss(similarity)
        else:
            loss = get_softmax_loss(similarity)

        return loss

# torch.manual_seed(0)
# feature_dim = 64 if config_param.key_word else 256
# embedding = torch.randn(64*10, feature_dim)
# Loss = GE2ELoss()
# loss = Loss(embedding)
# print(loss)

