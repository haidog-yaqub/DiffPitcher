# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.

from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
from scipy.io.wavfile import write
from .env import AttrDict
from .utils import MAX_WAV_VALUE
from .models import BigVGAN as Generator
import librosa


def load_model(model_path, device='cuda'):
    config_file = os.path.join(os.path.split(model_path)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)

    h = AttrDict(json_config)

    generator = Generator(h).to(device)

    cp_dict = torch.load(model_path, map_location=device)
    generator.load_state_dict(cp_dict['generator'])
    generator.eval()
    generator.remove_weight_norm()
    del cp_dict
    return generator, h

