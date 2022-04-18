import json
import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from collections import defaultdict




def read_tsv(file: str) -> Dict[str,List]:
    """read a .tsv file and return a dict.

    :param file: file to read from
    :return dataset_as_dict: it's a dict where the keys are the sentences id
        and the values are lists where each element is a tuple containing a
        word and its label
    """
    with open(file, 'r', encoding="utf8") as f:
        file_as_list = list(f)
    dataset_as_dict = {}

    for elem in file_as_list:
        if elem.startswith("#\tid"):
            sentence_id = elem.strip().split("\t")[-1]
            dataset_as_dict[sentence_id] = []
            continue
        if elem == "\n":
            continue
        elem = elem.strip()
        word_and_label = elem.split("\t")
        dataset_as_dict[sentence_id].append(word_and_label)

    return dataset_as_dict


def write_label_on_file(input_file: str, output_file: str):
    """Write the labels dict to a json file.

    :param input_file: tsv file
    :param output_file: json file
    """
    assert output_file.endswith(".json"), f"The output file should be a .json file got {output_file}"
    labels = []
    with open(input_file, 'r', encoding="utf8") as f:
        file_as_list = list(f)
    dataset_as_dict = {}

    for elem in file_as_list:
        if elem.startswith("#\tid"):
            sentence_id = elem.strip().split("\t")[-1]
            dataset_as_dict[sentence_id] = []
            continue
        if elem == "\n":
            continue
        elem = elem.strip()
        word_and_label = elem.split("\t")
        label = word_and_label[1]
        if label not in labels:
            labels.append(label)
        dataset_as_dict[sentence_id].append(word_and_label)

    id_to_labels = {}
    labels_to_id = {}
    for i, elem in enumerate(labels):
        id_to_labels[i] = elem
        labels_to_id[elem] = i

    labels = {"label_to_id": labels_to_id,
              "id_to_label": id_to_labels}
    with open(output_file, "w") as outfile:
        json.dump(labels, outfile)
        
def load_labels(label_file: str) -> Tuple[Dict,Dict]:
    """Load the labels from file and return 2 dict.

    :param label_file: .json file with the labels
    :return label_to_id:
    :return id_to_label:
    """
    
    with open(label_file, "r") as in_file:
        labels = json.load(in_file)
    label_to_id = labels["label_to_id"]
    id_to_label = labels["id_to_label"]
    
    return label_to_id, id_to_label


def count_labels(in_file: str, normalized: bool = True, exclude: list=None) -> Dict:
    """Return the number of labels.

    :param in_file: tsv input file
    :param normalized: if True return the normalized number of occurrences
    :param exclude: list of labels to exclude
    :return label_count: dict with the number of labels
    """
    with open(in_file, 'r', encoding="utf8") as f:
        file_as_list = list(f)


    label_count = defaultdict(lambda: 0)
    total_token = 0
    for elem in file_as_list:
        if elem.startswith("#\tid"):
            continue
        if elem == "\n":
            continue
        elem = elem.strip()
        word_and_label = elem.split("\t")
        label = word_and_label[1]
        if exclude is not None and label in exclude:
            continue
        label_count[label] += 1
        total_token += 1
        
    if normalized:
        for elem in label_count:
            label_count[elem] = label_count[elem]/total_token

    return label_count
    
def plot_dict(distr: dict):
    """Plot a distribution given a dict
    """
    plt.bar(range(len(distr)), list(distr.values()), align='center')
    plt.xticks(range(len(distr)), list(distr.keys()))


def extract_sequences(sequence: List[Tuple[str,str]], seq_len, skip_len, centered= False,):
    """extract sequence of length seq_len from the original sentence.

    It also add padding if needed
    
    :param seq_len: if centered is true, for each word the sequence will have
        seq_len//2 words before and seq_len//2 words after it. Otherwise it will
        return a normal sequence of len = seq_len 
    :param skip_len: parameter that control the windows skip when creating the
        new sequences
    :return new_sequence: lst of list of sequences of size seq_len already padded
    :return new_labels: list of list with the labels for the new sequence
    """
    assert skip_len < seq_len, f"sequence_len = {seq_len}, skip_len = {skip_len}The length of the sequences should be smaller than the windows skip otherwise you lose informations. "
    new_sequence = []
    new_labels = []
    if centered:
        seq_len = seq_len//2
        for i in range(0,len(sequence), skip_len):
            new_sequence.append([])
            new_labels.append([])
            for j in range(i-seq_len,i+seq_len+1):
                if j < 0:
                    new_sequence[-1].append("<pad>")
                    new_labels[-1].append("PAD")
                elif j > len(sequence)-1:
                    new_sequence[-1].append("<pad>")
                    new_labels[-1].append("PAD")
                else:
                    new_sequence[-1].append(sequence[j][0])
                    new_labels[-1].append(sequence[j][1])
    else:
        # start from -1 since we add the padding
        for i in range(-1,len(sequence), skip_len):
            new_sequence.append([])
            new_labels.append([])
            # PAD IF THE SEQUENCE IS SMALL
            if i + seq_len > len(sequence):
                for j in range(seq_len):
                    if i == -1 and j==0:
                        # add padding to mark the beginning of the sentence
                        new_sequence[-1].append("<pad>")
                        new_labels[-1].append("PAD")
                    elif i + j < len(sequence):
                        new_sequence[-1].append(sequence[i+j][0])
                        new_labels[-1].append(sequence[i+j][1])
                    else:
                        new_sequence[-1].append("<pad>")
                        new_labels[-1].append("PAD")
                
            else:
                for j in range(seq_len):
                    if i == -1 and j==0:
                        # add padding to mark the beginning of the sentence
                        new_sequence[-1].append("<pad>")
                        new_labels[-1].append("PAD")
                    else:
                        new_sequence[-1].append(sequence[i+j][0])
                        new_labels[-1].append(sequence[i+j][1])
                    

    return new_sequence, new_labels      


def pretrained_feature_extractor(text: str, missing_words: defaultdict,
                                 word2vec_embed):
    """Use a pretrained embedding the text.
    
    :param text:  list of lists where inner list has len = seq_len
    :param missing_words: dictionary of unseen words in the embedding model
    :param world2vec_embed: embedding vector
    :return embedded sentences: final embedding of the sentence
    """
    
    embedding = torch.zeros((len(text), len(text[0]),word2vec_embed.vector_size))
    for i, sub_sentences in enumerate(text):
        for j,word in enumerate(sub_sentences):
            if word in word2vec_embed: 
                embedding[i][j] = torch.from_numpy(word2vec_embed[word])
            else:
                missing_words[word] += 1
                embedding[i][j] = torch.from_numpy(word2vec_embed["<unk>"])

    return embedding

def update_missing_words(text: str, missing_words: defaultdict,
                                 word2vec_embed):
    """Update the list of missing words given a text
    
    :param text:  list of lists where inner list are single words
    :param missing_words: dictionary of unseen words in the embedding model
    :param world2vec_embed: embedding vector
    """
    for word in text:
        if word[0] in word2vec_embed: 
            continue
        else:
            missing_words[word[0]] += 1