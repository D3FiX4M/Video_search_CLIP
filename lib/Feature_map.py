import torch
from lib import Preprocessing
import numpy as np
from lib.Model import model


def image_map_features(images):
    image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()

    image_input = torch.tensor(np.stack(images)).cuda()
    image_input -= image_mean[:, None, None]
    image_input /= image_std[:, None, None]

    with torch.no_grad():
        image_features = model.encode_image(image_input).float()

    return image_features


def text_map_features(text):
    tokenizer = Preprocessing.SimpleTokenizer()

    text_tokens = [tokenizer.encode(tok) for tok in text]

    description = torch.zeros(len(text_tokens), model.context_length.item(), dtype=torch.long)

    sot_token = tokenizer.encoder['<|startoftext|>']
    eot_token = tokenizer.encoder['<|endoftext|>']

    for i, tokens in enumerate(text_tokens):
        tokens = [sot_token] + tokens + [eot_token]
        description[i, :len(tokens)] = torch.tensor(tokens)

    description = description.cuda()

    with torch.no_grad():
        text_features = model.encode_text(description).float()

    return text_features
