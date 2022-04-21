import os
import time
import string
import nltk
import sklearn
import json
import numpy as np
import wandb
import random
import torch
import gensim.downloader
from torch import nn, optim
from nltk.corpus import stopwords
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from Trainer import Trainer
from utils import read_tsv, load_labels, count_labels, plot_dict, extract_sequences,\
    pretrained_feature_extractor, update_missing_words, extract_embedding
from Dataset import NamedEntityRecognitionDataset
from torch.utils.data import DataLoader
from CustomModel import BiLSTMModel, BiLSTMCRFModel
from evaluate import read_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


MAIN_DIR = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
DATA_DIR = os.path.join(MAIN_DIR, "data")
TRAIN_FILE = os.path.join(DATA_DIR, "train.tsv")
DEV_FILE = os.path.join(DATA_DIR, "dev.tsv")
MODEL_DIR = os.path.join(MAIN_DIR, "model")
LABELS_FILE = os.path.join(MODEL_DIR, "labels_to_id.json")
MISSING_WORDS_FILE = os.path.join(MODEL_DIR, "word2vec-google-news-300-missing_words.json")
WEIGHTS_DIR= os.path.join(MODEL_DIR, "weights")
if not os.path.exists(WEIGHTS_DIR):
    os.mkdir(WEIGHTS_DIR)
# =============================================================================
# STOPWORDS AND EMBEDDING
# =============================================================================
# downloaded in C:/Users/Francesco/gensim-data
# use glove-wiki during debugging since it's faster to load
word2vec_embed = gensim.downloader.load('glove-wiki-gigaword-50')
# word2vec_embed = gensim.downloader.load('word2vec-google-news-300')
embedding_size = word2vec_embed.vector_size
word2vec_embed.add_vector("<pad>", np.zeros(embedding_size))
unk_embedding = word2vec_embed.vectors.mean(axis=0)
word2vec_embed.add_vector("<unk>", unk_embedding)


if __name__ == "__main__":
    # training_data = read_tsv(TRAIN_FILE)
    dev_data_dict = read_tsv(DEV_FILE)
    label_to_id, id_to_label = load_labels(LABELS_FILE)
    labels = [label for label in label_to_id.keys()]
    # label_dist = count_labels(DEV_FILE, exclude=["O"])
    # plot_dict(label_dist)
    # =============================================================================
    # PARAMS
    # =============================================================================
    num_of_classes = len(label_to_id)
    num_of_labels = len(label_to_id)
    embedding_len = word2vec_embed.vector_size
    # glorious-sweep-1 params
    # seq_len = 20
    # seq_skip = 2
    # bilstm_hidden_size = 64
    # batch_size = 50
    dropout = 0.3501
    # lilac-sweep-3 params
    seq_len = 20
    seq_skip = 2
    bilstm_hidden_size = 256
    batch_size = 16
    dropout = 0.233

    # =============================================================================
    #  MODEL
    # =============================================================================
    bilstm_crf_model = BiLSTMCRFModel(embedding_len, num_of_labels, bilstm_hidden_size, True, word2vec_embed, id_to_label, device, dropout)
    # sequence, label = extract_sequences(training_data["0"], seq_len, seq_skip, True)
    # sequence_2, label_2 = extract_sequences(training_data["0"], seq_len, seq_skip, False)
    
    # =============================================================================
    # DATASETS
    # =============================================================================
    # Create a dataset with fixed batch size and fixed sequence length.
    # While the recurrent networks accept every kind of sequence length, during
    # training we need equal length sequences within a batch. That's why we
    # use padding
    pretrained_feature_extractor = extract_embedding
    training_data = NamedEntityRecognitionDataset(TRAIN_FILE, pretrained_feature_extractor,
                                    label_to_id,seq_len,seq_skip, word2vec_embed)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    valid_data = NamedEntityRecognitionDataset(DEV_FILE, pretrained_feature_extractor,
                               label_to_id,seq_len,seq_skip, word2vec_embed)
    

    


    NEW_WEIGHTS_DIR = os.path.join(WEIGHTS_DIR, "lilac-sweep-3")
    model_name = f"{bilstm_crf_model.name}_25.pt"
    # NEW_WEIGHTS_DIR = os.path.join(WEIGHTS_DIR, "glorious-sweep-1")
    # model_name = f"{bilstm_crf_model.name}_29.pt"
    if not os.path.exists(NEW_WEIGHTS_DIR):
        os.mkdir(NEW_WEIGHTS_DIR)
    
    bilstm_crf_model.load_weights(os.path.join(NEW_WEIGHTS_DIR, model_name))
    y_true = None
    y_predicted = None
    # to remove dropout
    bilstm_crf_model.eval()
    with torch.no_grad():

        for i, sample in enumerate(valid_data):
            inputs = sample[0].type(
                torch.float32).unsqueeze(0).to(device)
            true_labels = sample[1].unsqueeze(0).to(device)

            scores, predicted_labels = bilstm_crf_model(inputs)   
            # remove padding
            true_labels = true_labels[:, [n for n in range(len(predicted_labels[0]))]]
            predicted_labels = torch.FloatTensor(predicted_labels).squeeze(0)
            true_labels = true_labels.squeeze(0)
            true_labels = true_labels.to("cpu")
            predicted_labels.to("cpu")
            if i == 0:    
                y_true = true_labels
                y_predicted = predicted_labels
            else:
                y_true = torch.cat((y_true, true_labels))
                y_predicted = torch.cat((y_predicted, predicted_labels))
                
    cm = confusion_matrix(y_true, y_predicted)
    ConfusionMatrixDisplay.from_predictions(y_true, y_predicted,display_labels=labels,cmap='Blues',normalize="true")