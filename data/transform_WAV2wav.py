import glob
import os
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

from utils.parse_config import config_param
from sphfile import SPHFile

WAV_files = glob.glob(config_param.data.TI_SV_data.unprocessed_data[:-4] + ".WAV")
print(len(WAV_files))

for WAV_file in WAV_files:
    sph = SPHFile(WAV_file)
    txt_file = WAV_file[:-3] + "TXT"

    print("writing file ", WAV_file)
    # if not os.path.isfile(WAV_file):
    sph.write_wav(WAV_file)

##############################
# for unknown reason, sphfile will transform original file to a error file in Windows. Instead, I transform in Linux.
##############################