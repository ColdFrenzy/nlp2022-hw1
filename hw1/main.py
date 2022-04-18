import os
import time
import string
import nltk
import sklearn
import numpy as np
import gensim.downloader
from nltk.corpus import stopwords
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from utils import read_tsv, load_labels, count_labels, plot_dict, extract_sequences, pretrained_feature_extractor
from model import BiLSTMModel



MAIN_DIR = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
DATA_DIR = os.path.join(MAIN_DIR, "data")
TRAIN_FILE = os.path.join(DATA_DIR, "train.tsv")
DEV_FILE = os.path.join(DATA_DIR, "dev.tsv")
MODEL_DIR = os.path.join(MAIN_DIR, "model")
LABELS_FILE = os.path.join(MODEL_DIR, "labels_to_id.json")


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
embedding_size = word2vec_embed.vector_size
word2vec_embed.add_vector("<pad>", np.zeros(embedding_size))
unk_embedding = word2vec_embed.vectors.mean(axis=0)
word2vec_embed.add_vector("<unk>", unk_embedding)
# word2vec_embed = gensim.downloader.load('word2vec-google-news-300')
print(f"Time to download and load word2vec: {time.time()-word2vec_start}")



if __name__ == "__main__":
    training_data = read_tsv(TRAIN_FILE)
    dev_data = read_tsv(DEV_FILE)
    
    label_to_id, id_to_label = load_labels(LABELS_FILE)
    labels = [label for label in label_to_id.keys()]
    # label_dist = count_labels(DEV_FILE, exclude=["O"])
    # plot_dict(label_dist)
    num_of_classes = len(label_to_id)
    seq_len = 5
    seq_skip = 3
    # =============================================================================
    #  MODEL
    # =============================================================================
    model = BiLSTMModel(300, 256, 1, 12, False, [128,64,32])
    
    sequence, label = extract_sequences(training_data["0"], seq_len, seq_skip, True)
    sequence_2, label_2 = extract_sequences(training_data["0"], seq_len, seq_skip, False)
    
    missing_words = defaultdict(int)
    embedding = pretrained_feature_extractor(sequence_2, missing_words, word2vec_embed)
    # epochs = 100
    # for epoch in epochs:
    #     print(f"Epoch {epoch}/{epochs}")
    
    
    # y_true = np.random.randint(0,13, 100)
    # y_pred = np.random.randint(0,13, 100)
    # confusion_matrix(y_true, y_pred)
    # ConfusionMatrixDisplay.from_predictions(y_true, y_pred,display_labels=labels,cmap='Blues',normalize="true")