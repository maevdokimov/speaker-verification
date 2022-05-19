import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from nemo.collections.tts.helpers.helpers import get_mask_from_lengths
from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types.elements import AcousticEncodedRepresentation, LengthsType, LogitsType
from nemo.core.neural_types.neural_type import NeuralType

# Loss type resolver. Loss type ->
# (Do we normalize output matrix rows?, Do we normalize embeddings before last layer?)
LOSS_TYPES = {"CE": (False, False), "AS": (True, False), "AAS": (True, True)}


class NormalizedLinear(nn.Module):
    """
    Linear layer with row normalization of projection matrix
    Args:
        in_features (int): size of each input sample
        out_features (int): size of each output sample
        bias (bool): if set to False, the layer will not learn an additive bias
        normalize_input (bool): whether to normalize input features.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool, normalize_input: bool):
        super().__init__()

        self.normalize_input = normalize_input
        self.bias = bias

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)

        normalized_weight = F.normalize(self.weight, p=2, dim=1)

        return F.linear(x, normalized_weight, self.bias)


class AttentivePoolingLayer(NeuralModule):
    def __init__(self, input_dim: int, intermediate_dim: int, num_classes: int, loss_type: str):
        super().__init__()

        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.normalize_emb, self.normalize_weight = LOSS_TYPES[loss_type]

        self.attention = nn.Sequential(
            nn.Conv1d(input_dim, intermediate_dim, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(intermediate_dim),
            nn.Conv1d(intermediate_dim, input_dim, kernel_size=1),
        )
        self.emb_layer = nn.Sequential(nn.Linear(input_dim, input_dim), nn.BatchNorm1d(input_dim), nn.ReLU())

        if self.normalize_weight:
            self.out_linear = NormalizedLinear(input_dim, num_classes, bias=False, normalize_input=self.normalize_emb)
        else:
            self.out_linear = nn.Linear(input_dim, num_classes)

    @property
    def input_types(self):
        return {
            "encoder_outputs": NeuralType(("B", "D", "T"), AcousticEncodedRepresentation()),
            "encoder_lengths": NeuralType(tuple("B"), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "logits": NeuralType(("B", "D"), LogitsType()),
            "embs": NeuralType(("B", "D"), AcousticEncodedRepresentation()),
        }

    @typecheck()
    def forward(self, encoder_outputs, encoder_lengths):
        attention_logits = self.attention(encoder_outputs)

        mask = ~get_mask_from_lengths(encoder_lengths, max_len=encoder_outputs.shape[2])
        mask = mask.unsqueeze(1)
        masked_logits = torch.masked_fill(attention_logits, mask, -1e9)
        attention_mask = F.softmax(masked_logits, dim=-1)
        attention_mask = torch.masked_fill(attention_mask, mask, 0.0)

        pool = torch.sum(attention_logits * attention_mask, dim=-1)
        pool = pool.squeeze()

        pre_norm_emb = self.emb_layer(pool)
        logits = self.out_linear(pre_norm_emb)

        return logits, pre_norm_emb


class LDEPoolingLayer(NeuralModule):
    def __init__(self, input_dim: int, num_clusters: int, num_classes: int, loss_type: str):
        super().__init__()

        self.input_dim = input_dim
        self.num_clusters = num_clusters
        self.num_classes = num_classes
        self.normalize_emb, self.normalize_weight = LOSS_TYPES[loss_type]

        self.dictionary_components = nn.Parameter(torch.empty(num_clusters, input_dim))
        nn.init.uniform_(self.dictionary_components, -1, 1)

        self.embed_proj = nn.Linear(input_dim * num_clusters, input_dim)
        if self.normalize_weight:
            self.out_linear = NormalizedLinear(input_dim, num_classes, bias=False, normalize_input=self.normalize_emb)
        else:
            self.out_linear = nn.Linear(input_dim, num_classes)

    @property
    def input_types(self):
        return {
            "encoder_outputs": NeuralType(("B", "D", "T"), AcousticEncodedRepresentation()),
            "encoder_lengths": NeuralType(tuple("B"), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "logits": NeuralType(("B", "D"), LogitsType()),
            "embs": NeuralType(("B", "D"), AcousticEncodedRepresentation()),
        }

    @typecheck()
    def forward(self, encoder_outputs, encoder_lengths):
        outputs_transposed = encoder_outputs.transpose(1, 2)  # B x T x D

        r = outputs_transposed.unsqueeze(2) - self.dictionary_components  # B x T x n_clust x D
        r_norm = torch.norm(r, p=2, dim=-1)  # B x T x n_clust
        w = F.softmax(-r_norm, dim=-1).unsqueeze(3)  # B x T x n_clust x 1

        mask = ~get_mask_from_lengths(encoder_lengths, max_len=encoder_outputs.shape[2])
        mask = mask.view(*mask.shape, 1, 1)
        masked_w = torch.masked_fill(w, mask, 0.0)  # B x T x n_clust x 1

        masked_w = masked_w / (torch.sum(masked_w, dim=1, keepdim=True) + 1e-9)  # B x T x n_clust
        embeddings = torch.sum(masked_w * r, dim=1)  # B x n_clust x D
        embeddings = embeddings.reshape(embeddings.shape[0], -1)  # B x n_clust * D
        embeddings = self.embed_proj(embeddings)  # B x D

        logits = self.out_linear(embeddings)

        return logits, embeddings
