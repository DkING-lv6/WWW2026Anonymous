import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias

        return self.act(out)


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)


class MaxReadout(nn.Module):
    def __init__(self):
        super(MaxReadout, self).__init__()

    def forward(self, seq):
        return torch.max(seq, 1).values


class MinReadout(nn.Module):
    def __init__(self):
        super(MinReadout, self).__init__()

    def forward(self, seq):
        return torch.min(seq, 1).values


class WSReadout(nn.Module):
    def __init__(self):
        super(WSReadout, self).__init__()

    def forward(self, seq, query):
        query = query.permute(0, 2, 1)
        sim = torch.matmul(seq, query)
        sim = F.softmax(sim, dim=1)
        sim = sim.repeat(1, 1, 64)
        out = torch.mul(seq, sim)
        out = torch.sum(out, 1)
        return out


def random_structure_perturbation(adj_row, perturb_ratio=0.02):

    adj_row = adj_row.clone()
    num_nodes = adj_row.shape[0]
    num_perturb = int(num_nodes * perturb_ratio)


    connected = (adj_row > 0).nonzero(as_tuple=True)[0]
    disconnected = (adj_row == 0).nonzero(as_tuple=True)[0]


    if len(connected) > 0:
        drop_idx = connected[torch.randperm(len(connected))[:num_perturb]]
        adj_row[drop_idx] = 0

    if len(disconnected) > 0:
        add_idx = disconnected[torch.randperm(len(disconnected))[:num_perturb]]
        adj_row[add_idx] = 1

    return adj_row


def loss_anomaly_severity(emb_normal, emb_fake_abn, emb_adv_abn, margin=0.5):
    """
    emb_normal: [N, D]
    emb_fake_abn:  [N, D]
    emb_adv_abn: [N, D]

    """
    # L2
    emb_normal = F.normalize(emb_normal, p=2, dim=1)
    emb_fake_abn = F.normalize(emb_fake_abn, p=2, dim=1)
    emb_adv_abn = F.normalize(emb_adv_abn, p=2, dim=1)


    D_pos = torch.norm(emb_fake_abn - emb_normal, dim=1)
    D_neg = torch.norm(emb_adv_abn - emb_normal, dim=1)


    loss_pos = D_pos.mean()


    loss_neg = torch.clamp(margin + D_pos - D_neg, min=0).mean()


    loss = loss_pos + loss_neg
    return loss


class Model(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(Model, self).__init__()

        self.gcn1 = GCN(n_in, n_h, activation)
        self.gcn2 = GCN(n_h, n_h, activation)
        self.gcn3 = GCN(n_h, n_h, activation)
        self.fc1 = nn.Linear(n_h, int(n_h / 2), bias=False)
        self.fc2 = nn.Linear(int(n_h / 2), int(n_h / 4), bias=False)
        self.fc3 = nn.Linear(int(n_h / 4), 1, bias=False)
        self.act = nn.ReLU()
        self.fc4 = nn.Linear(n_h, n_h, bias=False)
        self.dropout = nn.Dropout(p=0.5)


    def forward(self, seq1, adj, normal_idx, train_flag, args, sparse=False):
        device = seq1.device
        h_1 = self.gcn1(seq1, adj, sparse)
        h_1 = self.dropout(h_1)
        emb = self.gcn2(h_1, adj, sparse)

        emb_con1 = None
        emb_combine = None
        emb_con_fake_abn = None
        emb_con_advanced_abn = None
        loss1 = None
        emb_normal = None


        neigh_adj1 = adj[0, normal_idx, :]
        emb_con1 = torch.mm(neigh_adj1, emb[0])
        emb_con1 = self.act(self.fc4(emb_con1))

        if train_flag:
            emb_normal = emb[:, normal_idx, :]
            noise = (torch.randn(emb_normal.size()) * args.var + args.mean)
            # noise = torch.randn(emb_normal.size()) * args.var + args.mean
            emb_normal = emb_normal + noise


            neigh_adj_normal = adj[0, normal_idx, :]  # [num_normal, N]

            emb_with_noise = (emb[0] + torch.randn_like(emb[0]) * args.var + args.mean)  # shape: [N, dim]
            '''
            emb_with_noise = (emb[0] + (torch.rand_like(emb[0]) - 0.5) * 2 * args.var + args.mean).to(device)
            '''


            emb_con_fake_abn = torch.mm(neigh_adj_normal, emb_with_noise)  # shape: [num_normal, dim]

            emb_con_fake_abn = self.act(self.fc4(emb_con_fake_abn))  # shape: [num_normal, new_dim]


            perturbed_adj = []
            for i in range(neigh_adj_normal.shape[0]):
                perturbed_row = random_structure_perturbation(neigh_adj_normal[i], perturb_ratio=0.02)
                perturbed_adj.append(perturbed_row.unsqueeze(0))
            perturbed_adj = torch.cat(perturbed_adj, dim=0)  # shape [num_normal, N]

            emb_con_advanced_abn = torch.mm(perturbed_adj, emb_with_noise)
            emb_con_advanced_abn = self.act(self.fc4(emb_con_advanced_abn))


            emb_combine = torch.cat(
                (emb[:, normal_idx, :], torch.unsqueeze(emb_con_fake_abn, 0)), 1)


            emb_combine1 = torch.cat(
                (emb[:, normal_idx, :], torch.unsqueeze(emb_con_advanced_abn, 0)), 1)
            f_1 = self.fc1(emb_combine)
            f_1 = self.act(f_1)
            f_2 = self.fc2(f_1)
            f_2 = self.act(f_2)
            f_2 = self.dropout(f_2)
            f_3 = self.fc3(f_2)

            f_4 = self.fc1(emb_combine1)
            f_4 = self.act(f_4)
            f_5 = self.fc2(f_4)
            f_5 = self.act(f_5)
            f_5 = self.dropout(f_5)
            f_6 = self.fc3(f_5)

            loss1 = loss_anomaly_severity(emb_con1, emb_con_fake_abn, emb_con_advanced_abn, margin=1)
            return emb, emb_combine, f_3, f_6, emb_con1, emb_normal, emb_con_fake_abn, emb_con_advanced_abn, loss1
        else:
            f_1 = self.fc1(emb)
            f_1 = self.act(f_1)
            f_2 = self.fc2(f_1)
            f_2 = self.act(f_2)
            f_2 = self.dropout(f_2)
            f_3 = self.fc3(f_2)

            return emb, emb_combine, f_3, emb_con1, emb_normal, emb_con_fake_abn, emb_con_advanced_abn, loss1

