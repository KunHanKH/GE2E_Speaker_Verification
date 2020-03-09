import json
import os
import pickle as pkl

import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from utils.parse_config import config_param

def retrieve_key_word_data(json_path, data_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)
    speaker = dict()
    for data in data:
        if data['is_hotword']:
            if not speaker.get(data['worker_id']):
                speaker[data['worker_id']] = []
            speaker[data['worker_id']].append(os.path.join(data_dir, data['audio_file_path']))
    return speaker


if __name__ == "__main__":

    # set Snips data directory
    Snips_path = config_param.data.TD_SV_data.unprocessed_dir
    Snips_train = os.path.join(Snips_path, 'train')
    Snips_dev = os.path.join(Snips_path, 'dev')
    Snips_test = os.path.join(Snips_path, 'test')


    os.makedirs(Snips_path, exist_ok=True)
    os.makedirs(Snips_train, exist_ok=True)
    os.makedirs(Snips_dev, exist_ok=True)
    os.makedirs(Snips_test, exist_ok=True)

    # load json file
    with open('../config/Snips_config.json') as f:
        data_config = json.load(f)

    # retrieve key_word stat for train, dev and test
    train_speaker = retrieve_key_word_data(data_config["train_data_path"], data_config["data_dir"])
    dev_speaker = retrieve_key_word_data(data_config["dev_data_path"], data_config["data_dir"])
    test_speaker = retrieve_key_word_data(data_config["test_data_path"], data_config["data_dir"])

    # dump the retrieved stat
    pkl.dump(train_speaker, open(os.path.join(Snips_path, 'train_speaker.pkl'), 'wb'))
    pkl.dump(dev_speaker, open(os.path.join(Snips_path, 'dev_speaker.pkl'), 'wb'))
    pkl.dump(test_speaker, open(os.path.join(Snips_path, 'test_speaker.pkl'), 'wb'))

    print("there are total {} speaker in train set".format(len(train_speaker)))
    print("there are total {} speaker in dev set".format(len(dev_speaker)))
    print("there are total {} speaker in test set".format(len(test_speaker)))


    # extract key word train_data
    for speaker_id, utt_paths in train_speaker.items():
        speaker_dir = os.path.join(Snips_train, speaker_id)
        os.makedirs(speaker_dir, exist_ok=True)
        for path in utt_paths:
            os.system("cp {} {}".format(path, speaker_dir))
            print("Successfully copy: {}".format(path))

    # extract key word dev_data
    for speaker_id, utt_paths in dev_speaker.items():
        speaker_dir = os.path.join(Snips_dev, speaker_id)
        os.makedirs(speaker_dir, exist_ok=True)
        for path in utt_paths:
            os.system("cp {} {}".format(path, speaker_dir))
            print("Successfully copy: {}".format(path))

    # extract key word test_data
    for speaker_id, utt_paths in test_speaker.items():
        speaker_dir = os.path.join(Snips_test, speaker_id)
        os.makedirs(speaker_dir, exist_ok=True)
        for path in utt_paths:
            os.system("cp {} {}".format(path, speaker_dir))
            print("Successfully copy: {}".format(path))
