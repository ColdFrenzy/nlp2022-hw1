import os
import time
import string
import nltk
import sklearn
import json
import numpy as np
import wandb
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
# stopwords C:\Users\Francesco\AppData\Roaming\nltk_data
stopword_start = time.time()
nltk.download('stopwords')
stopset = set(stopwords.words('english') +
              [p for p in string.punctuation])
print(f"Time to download and load the stopset: {time.time()-stopword_start}")
word2vec_start = time.time()
# downloaded in C:/Users/Francesco/gensim-data
# use glove-wiki during debugging since it's faster to load
word2vec_embed = gensim.downloader.load('glove-wiki-gigaword-50')
# word2vec_embed = gensim.downloader.load('word2vec-google-news-300')
embedding_size = word2vec_embed.vector_size
word2vec_embed.add_vector("<pad>", np.zeros(embedding_size))
unk_embedding = word2vec_embed.vectors.mean(axis=0)
word2vec_embed.add_vector("<unk>", unk_embedding)
print(f"Time to download and load word2vec: {time.time()-word2vec_start}")


wandb.login()

# Pass your defaults to wandb.init
wandb.init(project="nlp2022-hw1")



if __name__ == "__main__":
    # training_data = read_tsv(TRAIN_FILE)
    # dev_data = read_tsv(DEV_FILE)
    label_to_id, id_to_label = load_labels(LABELS_FILE)
    labels = [label for label in label_to_id.keys()]
    # label_dist = count_labels(DEV_FILE, exclude=["O"])
    # plot_dict(label_dist)
    # =============================================================================
    # PARAMS
    # =============================================================================
    num_of_classes = len(label_to_id)
    seq_len = 5
    seq_skip = 1
    bilstm_hidden_size = 256
    embedding_len = word2vec_embed.vector_size
    num_of_labels = len(label_to_id)
    batch_size = 20
    lr = 0.0003
    alpha = 0.99
    eps = 1e-5
    # =============================================================================
    #  MODEL
    # =============================================================================
    bilstm_crf_model = BiLSTMCRFModel(embedding_len, num_of_labels, bilstm_hidden_size, True,device)
    # sequence, label = extract_sequences(training_data["0"], seq_len, seq_skip, True)
    # sequence_2, label_2 = extract_sequences(training_data["0"], seq_len, seq_skip, False)
    
    # =============================================================================
    # DATASETS
    # =============================================================================
    pretrained_feature_extractor = extract_embedding
    training_data = NamedEntityRecognitionDataset(TRAIN_FILE, pretrained_feature_extractor,
                                    label_to_id,seq_len,seq_skip, stopset, word2vec_embed)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    dev_data = NamedEntityRecognitionDataset(DEV_FILE, pretrained_feature_extractor,
                               label_to_id,seq_len,seq_skip, stopset, word2vec_embed)
    # =============================================================================
    # TRAINER
    # =============================================================================
    # loss is already computed inside the model
    optimizer = optim.Adam(bilstm_crf_model.parameters(), lr=lr, eps=eps)
    trainer = Trainer(bilstm_crf_model, optimizer, device)
    
    # =============================================================================
    # CHECK MISSING WORDS FROM THE DATASET AND WRITE THEM ON FILE
    # =============================================================================
    # missing_words_google = defaultdict(lambda: 0)
    # for sentence_id, sentence in training_data.items():
    #     update_missing_words(sentence, missing_words_google, word2vec_embed)
    # missing_words_google = dict(missing_words_google)
    # with open(MISSING_WORDS_FILE, 'w') as f:
    #     f.write(json.dumps(missing_words))
    
    # check if characters are only numeric isnumeric
    # check if they start with capital letter NO, they are already lowercased
    # check if a character is a floating point number r'' raw string
    # if re.match(r'^-?\d+(?:\.\d+)$', element)
    
    
    epochs = 10
    for epoch in range(epochs):
        avg_epoch_loss, valid_loss = trainer.train(train_dataloader,
                                                   dev_data, 1)
        to_log = {"train_loss": avg_epoch_loss,
                  "val_loss": valid_loss}
        wandb.log(to_log)
        model_name = f"{bilstm_crf_model.name}_{epoch}.pt"
        torch.save(bilstm_crf_model.state_dict(), os.path.join(WEIGHTS_DIR, model_name))
        
    
    
    # y_true = np.random.randint(0,13, 100)
    # y_pred = np.random.randint(0,13, 100)
    # confusion_matrix(y_true, y_pred)
    # ConfusionMatrixDisplay.from_predictions(y_true, y_pred,display_labels=labels,cmap='Blues',normalize="true")