import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from nemo.collections.tts.helpers.helpers import get_mask_from_lengths
from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types.elements import AcousticEncodedRepresentation, LengthsType, LogitsType
from nemo.core.neural_types.neural_type import NeuralType


class AttentivePoolingLayer(NeuralModule):
    def __init__(self, input_dim: int, intermediate_dim: int, num_classes: int, angular: bool):
        super().__init__()

        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.angular = angular

        self.attention = nn.Sequential(
            nn.Conv1d(input_dim, intermediate_dim, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(intermediate_dim),
            nn.Conv1d(intermediate_dim, input_dim, kernel_size=1),
        )
        self.emb_layer = nn.Sequential(nn.Linear(input_dim, input_dim), nn.BatchNorm1d(input_dim), nn.ReLU())

        self.out_weight = nn.Parameter(torch.empty(num_classes, input_dim))
        nn.init.kaiming_uniform_(self.out_weight, a=math.sqrt(5))
        if not angular:
            self.out_bias = nn.Parameter(torch.empty(num_classes))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.out_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.out_bias, -bound, bound)

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

        if self.angular:
            normalized_emb = F.normalize(pre_norm_emb, p=2, dim=1)
            normalized_weight = F.normalize(self.out_weight, p=2, dim=1)

            logits = F.linear(normalized_emb, normalized_weight)
        else:
            logits = F.linear(pre_norm_emb, self.out_weight, self.out_bias)

        return logits, pre_norm_emb


class LDEPoolingLayer(NeuralModule):
    def __init__(self, input_dim: int, num_clusters: int, num_classes: int, angular: bool):
        super().__init__()

        self.input_dim = input_dim
        self.num_clusters = num_clusters
        self.angular = angular
        self.num_classes = num_classes

        self.dictionary_components = nn.Parameter(torch.empty(num_clusters, input_dim))
        nn.init.uniform_(self.dictionary_components, -1, 1)

        self.out_weight = nn.Parameter(torch.empty(num_classes, input_dim))
        nn.init.kaiming_uniform_(self.out_weight, a=math.sqrt(5))
        if not angular:
            self.out_bias = nn.Parameter(torch.empty(num_classes))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.out_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.out_bias, -bound, bound)

        self.embed_proj = nn.Linear(input_dim * num_clusters, input_dim)

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
        r_norm = F.normalize(r, p=2, dim=-1)  # B x T x n_clust
        w = F.softmax(-r_norm, dim=-1).unsqueeze(1)  # B x T x n_clust x 1

        mask = ~get_mask_from_lengths(encoder_lengths, max_len=encoder_outputs.shape[2])
        mask = mask.view(*mask.shape, 1, 1)
        masked_w = torch.masked_fill(w, mask, 0.0)  # B x T x n_clust x 1

        masked_w = masked_w / (torch.sum(masked_w, dim=1, keepdim=True) + 1e-9)  # B x T x n_clust
        embeddings = torch.sum(masked_w * r, dim=1)  # B x n_clust x D
        embeddings = torch.reshape(embeddings.shape[0], -1)  # B x n_clust * D
        embeddings = self.embed_proj(embeddings)  # B x D

        if self.angular:
            normalized_emb = F.normalize(embeddings, p=2, dim=1)
            normalized_weight = F.normalize(self.out_weight, p=2, dim=1)

            logits = F.linear(normalized_emb, normalized_weight)
        else:
            logits = F.linear(embeddings, self.out_weight, self.out_bias)

        return logits, embeddings
