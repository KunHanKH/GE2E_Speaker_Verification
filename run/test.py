import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
import random
import time
import argparse

if os.path.join(os.path.dirname(__file__),'..') not in sys.path:
    sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from utils.parse_config import config_param
from data.data_load import SpeechDataset
from model.net import SpeechEmbedder
from model.loss import GE2ELoss
from utils.utils import get_similarity_eva, get_EER

def test(model_path):

    if not config_param.key_word:
        data_config = config_param.data.TI_SV_data
        model_config = config_param.model.TI_SV_model
        test_config = config_param.test.TI_SV_test
    else:
        data_config = config_param.data.TD_SV_data
        model_config = config_param.model.TD_SV_model
        test_config = config_param.test.TD_SV_test

    if model_path == None:
        model_path = test_config.model_path

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load test dataset and dataloader
    test_set = SpeechDataset()
    test_loader = DataLoader(test_set, batch_size=test_config.N, shuffle=False, drop_last=True)

    # construct model
    speech_embedder = SpeechEmbedder().to(device)
    ge2e_loss = GE2ELoss().to(device)

    # restore model
    speech_embedder.load_state_dict(torch.load(model_path))
    print("successfully load model")
    # speech_embedder.load_state_dict(torch.load(model_path)['speech_embedder'])
    # ge2e_loss.load_state_dict(torch.load(model_path)['ge2e_loss'])

    os.makedirs(os.path.dirname(test_config.EER_log_file), exist_ok=True)

    speech_embedder.eval()
    ge2e_loss.eval()

    avg_EER = 0
    batch_avg_EER_log = []

    for e in range(test_config.epochs):
        # Because dataloader drop last batch, so we shuffle all data in case some data will never be used
        test_set.shuffle()
        batch_avg_EER = 0
        for batch_id, batch in enumerate(test_loader):

            batch = batch.to(device)
            N, M, frames, nmels = batch.shape
            enrollment_batch, evaluation_batch = torch.split(batch, M//2, dim=1)
            enrollment_batch = enrollment_batch.reshape(N * M//2, frames, nmels)
            evaluation_batch = evaluation_batch.reshape(N * (M-M//2), frames, nmels)

            enrollment_embedding = speech_embedder(enrollment_batch)
            evaluation_embedding = speech_embedder(evaluation_batch)

            enrollment_embedding = enrollment_embedding.reshape(N, M//2, -1)
            evaluation_embedding = evaluation_embedding.reshape(N, M-M//2, -1)

            similarity = get_similarity_eva(enrollment_embedding, evaluation_embedding)

            EER, FRR, FAR, thresh =  get_EER(similarity)
            batch_avg_EER += EER
            print("\nepoch %d batch_id %d: EER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" % (e+1, batch_id+1, EER, thresh, FAR, FRR))

        batch_avg_EER = batch_avg_EER / (batch_id + 1)
        batch_avg_EER_log.append(batch_avg_EER)
        avg_EER += batch_avg_EER
    avg_EER = avg_EER / test_config.epochs

    EER_log = {"batch_avg_EER_log": batch_avg_EER_log, "avg_EER": avg_EER}

    torch.save(EER_log, test_config.EER_log_file)
    print("\n EER across {0} epochs: {1:.4f}".format(test_config.epochs, avg_EER))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, help="Restore model from this path.")
    args = parser.parse_args()

    test(args.model_path)

