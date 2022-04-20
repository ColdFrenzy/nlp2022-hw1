import torch
from torch.utils.data import Dataset
from typing import Callable, Dict
from collections import defaultdict
import json
from utils import read_tsv, extract_sequences, extract_labels


class NamedEntityRecognitionDataset(Dataset):
    def __init__(self,
                 dataset_path: str,
                 feature_extraction_function: Callable[[str], torch.tensor],
                 label_to_id: Dict,
                 seq_len: int,
                 skip_len: int,
                 word2vec_embed=None):
        """NER_Dataset init.

        :param dataset_path: tsv file
        :param feature_extraction_function:
        :param label_to_id:
        :param seq_len: number of elements in a single sample 
        :param skip_len:
        :param word2vec_embed:

        Returns
        -------
        None.

        """
        # standard constructor
        self.dataset_path = dataset_path
        self.label_to_id = label_to_id
        self.labels = [l for l in label_to_id.keys()]
        self.seq_len = seq_len
        self.skip_len = skip_len
        self.feature_extraction_function = feature_extraction_function
        # call to init the data
        self.missing_words = defaultdict(int)
        self.word2vec_embed = word2vec_embed
        self._init_data()

    def _init_data(self):
        # iterate on the given file and build samples
        self.samples = []

        dataset_as_dict = read_tsv(self.dataset_path)


        for sentence_id in dataset_as_dict:
            subsentences, sub_label = extract_sequences(dataset_as_dict[sentence_id],seq_len = self.seq_len, skip_len=self.skip_len, labels=self.labels,centered=False)
            for words, labels in zip(subsentences,sub_label):
                self.samples.append(
                    (self.feature_extraction_function(
                        words, self.word2vec_embed, self.missing_words),
                     extract_labels(labels,self.label_to_id)))

    def __len__(self):
        """Return the number of samples in our dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Return the idx-th sample."""
        return self.samples[idx]

