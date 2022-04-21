import torch
import numpy as np
from torch import argmax
from torch import nn
from typing import List
from collections import OrderedDict
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import pretrained_feature_extractor
from model import Model


class FCModel(nn.Module, Model):
    def __init__(self,
                 input_shape: int,
                 num_classes: int,
                 fc_layers: List[int],
                 dropout: float,
                 activ_fun=nn.ReLU,
                 device="cpu"
                 ):
        """A fully connected network

        :param input_shape : shape of the input tensors
        :param num_classes : number of classes
        :param fc_layers: a list containing the input
        :param activ_funct: a torch.nn activation class
        """
        assert len(fc_layers) > 0, "You need to define at least 1 hidden layer"
        super(FCModel, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        network = OrderedDict()
        network["fc_input_layer"] = nn.Linear(self.input_shape, fc_layers[0],device=device)
        network["fc_input_activation"] = activ_fun()
        for i in range(len(fc_layers)-1):
            network["fc_layers_" + str(i+1)] = nn.Linear(fc_layers[i],
                                                         fc_layers[i+1],device=device)
            network["fc_activ_" + str(i+1)] = activ_fun()
            network["fc_layer_"+str(i+1) +
                    "_dropout"] = torch.nn.Dropout(dropout)
        network["fc_output_layer"] = nn.Linear(fc_layers[-1], self.num_classes,device=device)
        network["fc_output_activ"] = nn.Softmax(dim=1)
        self.model = nn.Sequential(network)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Compute network output given an input.

        :param input_data: [batch_size, self.input_shape]
        :return classes_distrib: [batch_size, self.num_classes]
        """
        classes_distrib = self.model(input_data)
        return classes_distrib
    
    
class BiLSTMModel(nn.Module, Model):
    def __init__(self,
                 input_features: int,
                 hidden_size: int,
                 num_layers: int,
                 num_classes: int,
                 batch_first: bool,
                 fc_layers: List[int],
                 dropout: float = 0.0,
                 ):
        """A Bidirectional LSTM model with additional fc layers.
        :param input_features:
        :param hidden_size:
        :param num_layers: number of lstm to stack
        :param num_classes: 
        :param batch_first:
        :param fc_layers:
        :param dropout: 
        :param activ_fun:

        Returns
        -------
        None.

        """
        
        super(BiLSTMModel, self).__init__()
        self.name = "BiLSTM_model"
        self.bilstm = nn.LSTM(input_features, hidden_size, num_layers,dropout=dropout,
                            bidirectional=True, batch_first=batch_first)
        
        self.fc_model = FCModel(hidden_size*2, num_classes, fc_layers, dropout)
        
        
        
    def forward(self, input_data: torch.Tensor, last_only: bool=True) -> torch.Tensor:
        """Compute network output given an input.

        :param input_data: [batch_size, self.input_shape]
        :param last_only: if true return only the last lstm output
        :return classes_distrib: [batch_size, self.num_classes]
        """
        lstm_output,_ = self.bilstm(input_data)
        classes_distrib = self.fc_model(lstm_output)
        return classes_distrib
    

# BiLSTM-CRF from:
# https://github.com/jidasheng/bi-lstm-crf
# Re-adapted in order to use pretrained embedding. Furthermore
# all the data preprocessing (batching, padding etc) is done outside

def log_sum_exp(x):
    """calculate log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    """
    max_score = x.max(-1)[0]
    return max_score + (x - max_score.unsqueeze(-1)).exp().sum(-1).log()


IMPOSSIBLE = -1e4


class CRF(nn.Module):
    """General CRF module.
    :param in_features: number of features for the input
    :param num_tag: number of tags.
    as the last 2 labels.
    """

    def __init__(self, num_classes, device="cpu"):
        super(CRF, self).__init__()

        self.num_classes = num_classes + 2
        self.start_idx = self.num_classes - 2
        self.stop_idx = self.num_classes - 1
        # transition factor, Tij mean transition from j to i
        self.transitions = nn.Parameter(torch.randn(self.num_classes, self.num_classes,device=device), requires_grad=True)
        self.transitions.data[self.start_idx, :] = IMPOSSIBLE
        self.transitions.data[:, self.stop_idx] = IMPOSSIBLE

    def forward(self, features, masks):
        """decode tags
        :param features: [B, L, C], batch of unary scores
        :param masks: [B, L] masks
        :return: (best_score, best_paths)
            best_score: [B]
            best_paths: [B, L]
        """
        return self.__viterbi_decode(features, masks[:, :features.size(1)].float())

    def loss(self, features, ys, masks):
        """negative log likelihood loss
        B: batch size, L: sequence length, D: dimension
        :param features: [B, L, D]
        :param ys: tags, [B, L]
        :param masks: masks for padding, [B, L]
        :return: loss
        """

        L = features.size(1)
        masks_ = masks[:, :L].float()

        forward_score = self.__forward_algorithm(features, masks_)
        gold_score = self.__score_sentence(features, ys[:, :L].long(), masks_)
        loss = (forward_score - gold_score).mean()
        return loss

    def __score_sentence(self, features, tags, masks):
        """Gives the score of a provided tag sequence
        :param features: [B, L, C]
        :param tags: [B, L]
        :param masks: [B, L]
        :return: [B] score in the log space
        """
        B, L, C = features.shape

        # emission score
        emit_scores = features.gather(dim=2, index=tags.unsqueeze(-1)).squeeze(-1)

        # transition score
        start_tag = torch.full((B, 1), self.start_idx, dtype=torch.long, device=tags.device)
        tags = torch.cat([start_tag, tags], dim=1)  # [B, L+1]
        trans_scores = self.transitions[tags[:, 1:], tags[:, :-1]]

        # last transition score to STOP tag
        last_tag = tags.gather(dim=1, index=masks.sum(1).long().unsqueeze(1)).squeeze(1)  # [B]
        last_score = self.transitions[self.stop_idx, last_tag]

        score = ((trans_scores + emit_scores) * masks).sum(1) + last_score
        return score

    def __viterbi_decode(self, features, masks):
        """decode to tags using viterbi algorithm
        :param features: [B, L, C], batch of unary scores
        :param masks: [B, L] masks
        :return: (best_score, best_paths)
            best_score: [B]
            best_paths: [B, L]
        """
        B, L, C = features.shape

        bps = torch.zeros(B, L, C, dtype=torch.long, device=features.device)  # back pointers

        # Initialize the viterbi variables in log space
        max_score = torch.full((B, C), IMPOSSIBLE, device=features.device)  # [B, C]
        max_score[:, self.start_idx] = 0
        # The previous is an hard constraint and furthermore we don't always
        # use batch with full sentences, we may have piece of sentences within
        # a sentence
        # max_score = torch.zeros((B,C), device=features.device)

        for t in range(L):
            mask_t = masks[:, t].unsqueeze(1)  # [B, 1]
            # output of the BiLSTM for the Tth element of the sequence
            emit_score_t = features[:, t]  # [B, C]

            # [B, 1, C] + [C, C]
            acc_score_t = max_score.unsqueeze(1) + self.transitions  # [B, C, C]
            acc_score_t, bps[:, t, :] = acc_score_t.max(dim=-1)
            acc_score_t += emit_score_t
            max_score = acc_score_t * mask_t + max_score * (1 - mask_t)  # max_score or acc_score_t

        # Transition to STOP_TAG
        max_score += self.transitions[self.stop_idx]
        best_score, best_tag = max_score.max(dim=-1)

        # Follow the back pointers to decode the best path.
        best_paths = []
        bps = bps.cpu().numpy()
        for b in range(B):
            best_tag_b = best_tag[b].item()
            seq_len = int(masks[b, :].sum().item())

            best_path = [best_tag_b]
            for bps_t in reversed(bps[b, :seq_len]):
                best_tag_b = bps_t[best_tag_b]
                best_path.append(best_tag_b)
            # drop the last tag and reverse the left
            best_paths.append(best_path[-2::-1])

        return best_score, best_paths

    def __forward_algorithm(self, features, masks):
        """calculate the partition function with forward algorithm.
        TRICK: log_sum_exp([x1, x2, x3, x4, ...]) = log_sum_exp([log_sum_exp([x1, x2]), log_sum_exp([x3, x4]), ...])
        :param features: features. [B, L, C]
        :param masks: [B, L] masks
        :return:    [B], score in the log space
        """
        B, L, C = features.shape

        scores = torch.full((B, C), IMPOSSIBLE, device=features.device)  # [B, C]
        scores[:, self.start_idx] = 0.
        scores = torch.zeros((B,C), device=features.device)
        trans = self.transitions.unsqueeze(0)  # [1, C, C]

        # Iterate through the sentence
        for t in range(L):
            emit_score_t = features[:, t].unsqueeze(2)  # [B, C, 1]
            score_t = scores.unsqueeze(1) + trans + emit_score_t  # [B, 1, C] + [1, C, C] + [B, C, 1] => [B, C, C]
            score_t = log_sum_exp(score_t)  # [B, C]

            mask_t = masks[:, t].unsqueeze(1)  # [B, 1]
            scores = score_t * mask_t + scores * (1 - mask_t)
        scores = log_sum_exp(scores) # + self.transitions[self.stop_idx])
        return scores
   
class BiLSTMCRFModel(nn.Module, Model):
    def __init__(self,
                 input_features: int,
                 num_classes: int,
                 hidden_size: int,
                 batch_first: bool,
                 embedding_dict: "gensim pretrained embedding",
                 id_to_labels: dict,
                 device= "cpu",
                 dropout = 0.0,
                 ):
        super(BiLSTMCRFModel, self).__init__()
        self.name = "BiLSTM_crf_model"
        self.num_classes = num_classes
        self.embedding_dict = embedding_dict
        self.device = device
        self.bilstm = nn.LSTM(input_features, hidden_size//2,bidirectional=True,batch_first=batch_first,device=device, dropout=dropout)
        self.id_to_labels = id_to_labels
        self.hidden_size = (hidden_size//2)*2
        self.fc_layer = nn.Linear(self.hidden_size,num_classes,device=device)
        
        self.crf_layer = CRF(num_classes, device)
    

    def extract_features(self, inputs):
        """Return the bilstm output and the padding mask.
        :param inputs: already padded tensor of shape
            [BATCH_SIZE, MAX_SEQ_LEN, FEATURE_LEN]
        """
        B, M, F = inputs.shape
        pad_tensor = torch.zeros(F,device=inputs.device)
        masks = torch.full((B,M), True, device=inputs.device)
        for b in range(B):
            for m in range(M):
                if inputs[b][m].equal(pad_tensor):
                    masks[b][m] = False
                    
        # sum (number of True values) along the first axis of the masks tensor
        seq_length = masks.sum(1)

        bilstm_output, _ = self.bilstm(inputs)

        return bilstm_output, masks

    def to(self, device):
        """Move model to device
        """
        self.bilstm.to(device)
        self.fc_layer.to(device)
        self.crf_layer.to(device)
        
        return self

    def loss(self, xs, tags):
        features, masks = self.extract_features(xs)
        features = self.fc_layer(features)
        loss = self.crf_layer.loss(features, tags, masks=masks)
        return loss

    def forward(self, inputs):
        # Get the emission scores from the BiLSTM
        features, masks = self.extract_features(inputs)
        features = self.fc_layer(features)
        scores, tag_seq = self.crf_layer(features, masks)
        return scores, tag_seq
    
    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
  
        embedded_tokens = pretrained_feature_extractor(tokens, self.embedding_dict).to(self.device)
        with torch.no_grad():
            scores, tag_seq = self.forward(embedded_tokens)
        predictions = []
        for sentence_labels in tag_seq:
            predictions.append([])
            for label in sentence_labels:
                predictions[-1].append(self.id_to_labels[str(label)])
                
        return predictions
        
    def load_weights(self, path_to_weight: str):
        """load weights from a .pt file

        """
        self.load_state_dict(torch.load(path_to_weight)) 

        
        
        
if __name__ == "__main__":
    import gensim.downloader
    # =============================================================================
    # STOPWORDS AND EMBEDDING
    # =============================================================================
    # downloaded in C:/Users/Francesco/gensim-data
    # use glove-wiki during debugging since it's faster to load
    device = "cpu"
    word2vec_embed = gensim.downloader.load('glove-wiki-gigaword-50')
    # word2vec_embed = gensim.downloader.load('word2vec-google-news-300')
    embedding_size = word2vec_embed.vector_size
    word2vec_embed.add_vector("<pad>", np.zeros(embedding_size))
    unk_embedding = word2vec_embed.vectors.mean(axis=0)
    word2vec_embed.add_vector("<unk>", unk_embedding)
    label_to_id = {"O": 0, "B-LOC": 1, "B-CW": 2, "I-CW": 3, "B-PER": 4, "I-PER": 5, "B-CORP": 6, "I-CORP": 7, "B-GRP": 8, "I-GRP": 9, "B-PROD": 10, "I-PROD": 11, "I-LOC": 12}
    id_to_label = {"0": "O", "1": "B-LOC", "2": "B-CW", "3": "I-CW", "4": "B-PER", "5": "I-PER", "6": "B-CORP", "7": "I-CORP", "8": "B-GRP", "9": "I-GRP", "10": "B-PROD", "11": "I-PROD", "12": "I-LOC"}
    # id_to_label[str(len(id_to_label))] = "START_TAG"
    # id_to_label[str(len(id_to_label))] = "STOP_TAG"
    # label_to_id["START_TAG"] = len(label_to_id)
    # label_to_id["STOP_TAG"] = len(label_to_id)
    bilstm_model = BiLSTMModel(50, 256, 1, 12, True, [128,64,32])
    bilstm_crf_model = BiLSTMCRFModel(50, len(label_to_id), 256, True, word2vec_embed, id_to_label, device)
    # We need batch first in order to easily create the masks.
    # BATCH_FIRST=FALSE -> [SEQ_LEN, BATCH_SIZE, FEATURES]
    # BATCH_FIRST=TRUE -> [BATCH_SIZE, SEQ_LEN, FEATURES]
    fake_input = torch.rand([10,5,50])
    fake_label = torch.randint(len(id_to_label), (10,5))
    pad_tensor = torch.zeros(50)
    # zero out some to simulated the
    fake_input[0][5:5] = pad_tensor
    fake_input[2][2:5] = pad_tensor
    fake_input[4][3:5] = pad_tensor
    fake_input[5][4:5] = pad_tensor
    fake_input[6][3:5] = pad_tensor
    fake_input[7][4:5] = pad_tensor
    fake_input[8][4:5] = pad_tensor
    fake_input[9][1:5] = pad_tensor
    bilstm_output = bilstm_model(fake_input)
    score, label_sequence = bilstm_crf_model(fake_input)
    loss = bilstm_crf_model.loss(fake_input, fake_label)
    sentence_to_predict = [["My", "name", "is", "Robin", "Hood"], ["Hey", "how","are","you"]]
    predictions = bilstm_crf_model.predict(sentence_to_predict)


    