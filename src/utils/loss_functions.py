import math
import torch
import torch.nn.functional as F

def local_global_loss_(l_enc, g_enc, batch):
    device = l_enc.device

    num_graphs = g_enc.size(0)
    num_nodes = l_enc.size(0)
    pos_mask = torch.zeros(num_nodes, num_graphs, device=device)
    neg_mask = torch.ones(num_nodes, num_graphs, device=device)

    for node_idx, graph_idx in enumerate(batch):
        pos_mask[node_idx][graph_idx] = 1.0
        neg_mask[node_idx][graph_idx] = 0.0

    scores = torch.mm(l_enc, g_enc.t())

    log_2 = math.log(2.)
    # p_samples = res * pos_mask
    # q_samples = res * neg_mask

    Ep = log_2 - F.softplus(-scores)
    Ep = Ep * pos_mask
    Eq = F.softplus(-scores) + scores - log_2
    Eq = Eq * neg_mask

    Ep_mean = Ep.sum() / num_nodes
    Eq_mean = Eq.sum() / (num_nodes * (num_graphs - 1))

    return Eq_mean - Ep_mean