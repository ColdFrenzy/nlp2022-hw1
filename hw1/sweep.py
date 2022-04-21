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
import random
import gensim.downloader
from torch import nn, optim
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from Trainer import Trainer
from utils import load_labels, pretrained_feature_extractor, extract_embedding
from Dataset import NamedEntityRecognitionDataset
from torch.utils.data import DataLoader
from CustomModel import BiLSTMCRFModel

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

random.seed(42)

wandb.login()

hyperparams_defaults = dict(
    dropout=0.2,
    hidden_size=128,
    learn_rate=0.01,
    batch_size=16,
    seq_len=5,
    seq_skip=3,
    epochs=100,
)

# Pass your defaults to wandb.init
wandb.init(config=hyperparams_defaults, project="nlp2022-hw1")
# Access all hyperparameter values through wandb.config
config = wandb.config
run_name = wandb.run.name



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
    seq_len = config.seq_len
    seq_skip = config.seq_skip
    bilstm_hidden_size = config.hidden_size
    embedding_len = word2vec_embed.vector_size
    num_of_labels = len(label_to_id)
    batch_size = config.epochs
    lr = config.learn_rate
    dropout = config.dropout
    alpha = 0.99
    eps = 1e-5
    # =============================================================================
    #  MODEL
    # =============================================================================
    bilstm_crf_model = BiLSTMCRFModel(embedding_len, num_of_labels, bilstm_hidden_size, True ,word2vec_embed, id_to_label,device, dropout)

    # sequence, label = extract_sequences(training_data["0"], seq_len, seq_skip, True)
    # sequence_2, label_2 = extract_sequences(training_data["0"], seq_len, seq_skip, False)
    
    # =============================================================================
    # DATASETS
    # =============================================================================
    pretrained_feature_extractor = extract_embedding
    training_data = NamedEntityRecognitionDataset(TRAIN_FILE, pretrained_feature_extractor,
                                    label_to_id,seq_len,seq_skip, word2vec_embed)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    dev_data = NamedEntityRecognitionDataset(DEV_FILE, pretrained_feature_extractor,
                               label_to_id,seq_len,seq_skip, word2vec_embed)
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
    
    
    epochs = 30
       
    for epoch in range(epochs):
        avg_epoch_loss, valid_loss, metrics = trainer.train(train_dataloader,
                                                    dev_data, 1)
        to_log = {"train_loss": avg_epoch_loss,
                  "valid_loss": valid_loss}
        to_log.update(metrics)
        wandb.log(to_log)
        NEW_WEIGHTS_DIR = os.path.join(WEIGHTS_DIR, run_name)
        if not os.path.exists(NEW_WEIGHTS_DIR):
            os.mkdir(NEW_WEIGHTS_DIR)
        model_name = f"{bilstm_crf_model.name}_{epoch}.pt"
        torch.save(bilstm_crf_model.state_dict(), os.path.join(NEW_WEIGHTS_DIR, model_name))
  


