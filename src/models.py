import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax
from torch_geometric.nn import GINConv, global_add_pool
from src.utils.loss_functions import local_global_loss_

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)
    
class Encoder(nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, attention_temperature=1):
        super().__init__()

        self.attention_temperature = attention_temperature

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.gate_nn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(dim, 1),
        )

        for layer_idx in range(num_gc_layers):
            input_dim = num_features if layer_idx == 0 else dim

            mlp = nn.Sequential(
                nn.Linear(input_dim, dim),
                nn.ReLU(),
                nn.Linear(dim, dim),
            )

            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(dim))

    def forward(self, x, edge_index, batch, pooling='attention'):
        layer_embeddings = []

        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(conv(x, edge_index))
            x = bn(x)
            layer_embeddings.append(x)

        attention_weights = None

        if pooling == "attention":
            pooled_embeddings = []
            attention_weights = []

            for layer_embedding in layer_embeddings:
                gate_logits = self.gate_nn(layer_embedding).view(-1, 1)

                gate_logits = (gate_logits / self.attention_temperature)

                alpha = softmax(gate_logits, batch)

                pooled_embedding = global_add_pool(alpha * layer_embedding, batch)

                attention_weights.append(alpha)
                pooled_embeddings.append(pooled_embedding)

        elif pooling == "sum":
            pooled_embeddings = [
                global_add_pool(layer_embedding, batch)
                for layer_embedding in layer_embeddings
            ]

        elif pooling is None:
            pooled_embeddings = layer_embeddings

        else:
            raise ValueError(
                f"Unsupported pooling method: {pooling}"
            )

        graph_embedding = torch.cat(pooled_embeddings, dim=1)
        node_embedding = torch.cat(layer_embeddings, dim=1)

        return (graph_embedding, node_embedding, attention_weights)

class InfoGraph(nn.Module):
    def __init__(self, num_features, hidden_dim, num_gc_layers, attention_temperature=1):
        super().__init__()

        self.embedding_dim = hidden_dim * num_gc_layers
        self.encoder = Encoder(num_features, hidden_dim, num_gc_layers, attention_temperature=attention_temperature)
        self.local_d = Discriminator(self.embedding_dim)
        self.global_d = Discriminator(self.embedding_dim)
        self.init_emb()
    
    def init_emb(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

                if module.bias is not None:
                    module.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch):

        graph_embedding, node_embedding, _ = self.encoder(x, edge_index, batch, pooling='attention')
        
        g_enc = self.global_d(graph_embedding)
        l_enc = self.local_d(node_embedding)

        loss = local_global_loss_(l_enc, g_enc, batch)
        
        return loss