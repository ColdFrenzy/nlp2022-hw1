import numpy as np
import os
import torch
from torch import nn
from typing import List, Tuple
from .CustomModel import  BiLSTMCRFModel, CRF
from model import Model
from .utils import load_labels
import gensim.downloader

MAIN_DIR = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir,os.pardir))
MODEL_DIR = os.path.join(MAIN_DIR, "model")
LABELS_FILE = os.path.join(MODEL_DIR, "labels_to_id.json")
WEIGHTS_DIR= os.path.join(MODEL_DIR, "weights")
WEIGHTS_FILE = os.path.join(MODEL_DIR, "model_weights.pt")
word2vec_embed = gensim.downloader.load('glove-wiki-gigaword-50')
embedding_size = word2vec_embed.vector_size
word2vec_embed.add_vector("<pad>", np.zeros(embedding_size))
unk_embedding = word2vec_embed.vectors.mean(axis=0)
word2vec_embed.add_vector("<unk>", unk_embedding)
embedding_len = word2vec_embed.vector_size
label_to_id, id_to_label = load_labels(LABELS_FILE)
num_of_labels = len(label_to_id)
# PARAMS FROM THE HYPERAPAMETER TUNING
bilstm_hidden_size = 256
dropout = 0.233

def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    # return RandomBaseline()
    model = StudentModel(device)
    model.load_weights(WEIGHTS_FILE)
    return model
    
class RandomBaseline(Model):
    options = [
        (3111, "B-CORP"),
        (3752, "B-CW"),
        (3571, "B-GRP"),
        (4799, "B-LOC"),
        (5397, "B-PER"),
        (2923, "B-PROD"),
        (3111, "I-CORP"),
        (6030, "I-CW"),
        (6467, "I-GRP"),
        (2751, "I-LOC"),
        (6141, "I-PER"),
        (1800, "I-PROD"),
        (203394, "O")
    ]

    def __init__(self):
        self._options = [option[1] for option in self.options]
        self._weights = np.array([option[0] for option in self.options])
        self._weights = self._weights / self._weights.sum()

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        return [
            [str(np.random.choice(self._options, 1, p=self._weights)[0]) for _x in x]
            for x in tokens
        ]


class StudentModel(Model):
    def __init__(self, device):
        self.model = BiLSTMCRFModel(embedding_len, num_of_labels, bilstm_hidden_size, True, word2vec_embed, id_to_label, device, dropout)

    def load_weights(self, path):
        self.model.load_weights(path)
    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of tokens!
        return self.model.predict(tokens)
