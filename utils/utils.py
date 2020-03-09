import numpy as np
import librosa
import torch
import torch.nn.functional as F

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

def get_centroid_include_self(embedding):
    '''
    calculate centroid embedding. For each embedding, include itself inside the calculation.
    :param embedding: shape -> (N, M, feature_dim)
    :return:
    embedding_mean: shape -> (M, feature_dim)
    '''
    N, M, feature_dim = embedding.shape
    embedding_mean = torch.mean(embedding, dim=1)
    return embedding_mean

def get_centroid_exclude_self(embedding):
    '''
    calculate centroid embedding. For each embedding, exclude itself inside the calculation.
    :param embedding: shape -> (N, M, feature_dim)
    :return:
    embedding_mean: shape -> (N, M, feature_dim)
    '''
    N, M, feature_dim = embedding.shape
    embedding_sum = torch.sum(embedding, dim=1, keepdim=True) # shape -> (N, 1, feature_dim)
    embedding_mean = (embedding_sum - embedding) / (M-1)
    return embedding_mean

def get_similarity_include_self(embedding, centroid_embedding):
    '''
    calculate similarity S_jik
    :param embedding: shape -> (N, M, feature_dim)
    :param centroid_embedding: -> (N, feature_dim)
    :return:
    similarity: shape -> (N, M, N)
    '''
    N, M, feature_dim = embedding.shape
    N_c, feature_dim_c = centroid_embedding.shape
    assert N == N_c and feature_dim == feature_dim_c, "dimension wrong in get_similarity_include_self!"

    centroid_embedding = centroid_embedding.unsqueeze(0).unsqueeze(0).expand(N, M, -1, -1)
    assert centroid_embedding.shape == (N, M, N, feature_dim), "centroid embedding has wrong expansion in get_similarity_include_self."
    embedding = embedding.unsqueeze(2)
    print(embedding.shape, centroid_embedding.shape)
    similarity = F.cosine_similarity(embedding, centroid_embedding, dim=3)
    return similarity

def get_similarity_exclude_self(embedding, centroid_embedding):
    '''
    calculate cimilarity S_jik for j == k
    :param embedding: shape -> (N, M, feature)
    :param centroid_embedding: shape -> (N, M, feature)
    :return:
    similarity: shape -> (N, M)
    '''
    N, M, feature_dim = embedding.shape
    N_c, M_c, feature_dim_c = centroid_embedding.shape
    assert N==N_c and M==M_c and feature_dim==feature_dim_c, "dimension wrong in get_similarity_exclude_self!"

    similarity = F.cosine_similarity(embedding, centroid_embedding, dim=2)
    return similarity

def get_similarity(embedding):
    '''
    get similarity for input embedding
    :param embedding: shape -> (N, M, feature)
    :return:
    similarity: shape -> (N, M, N)
    '''
    embedding_mean_include = get_centroid_include_self(embedding)
    embedding_mean_exclude = get_centroid_exclude_self(embedding)

    similarity_include = get_similarity_include_self(embedding, embedding_mean_include) # shape (N, M, N)
    similarity_exclude = get_similarity_exclude_self(embedding, embedding_mean_exclude) # shape (N, M)

    similarity = similarity_include
    same_index = list(range(similarity.shape[0]))
    similarity[same_index, :, same_index] = similarity_exclude[same_index, :]

    return similarity

def get_contrast_loss(similarity):
    '''
    L(e_ji) = 1-sigmoid(S_jij)+max_k(sigmoid(S_jik))
    :param similarity: shape -> (N, M, N)
    :return:
    loss = sum_ji(L(e_ji))
    '''
    sigmoid = torch.sigmoid(similarity)
    same_index = list(range(similarity.shape[0]))
    loss_1 = torch.sum(1-sigmoid[same_index, :, same_index])
    sigmoid[same_index, :, same_index] = 0
    loss_2 = torch.sum(torch.max(sigmoid, dim=2)[0])

    loss = loss_1 + loss_2
    return loss

def get_softmax_loss(similarity):
    '''
    L(e_ji) = -S_jij) + log(sum_k(exp(S_jik))
    :param similarity: shape -> (N, M, N)
    :return:
    loss = sum_ji(L(e_ji))
    '''
    same_index = list(range(similarity.shape[0]))
    loss = torch.sum(torch.log(torch.sum(torch.exp(similarity), dim=2) + 1e-6)) - torch.sum(similarity[same_index, :, same_index])
    return loss