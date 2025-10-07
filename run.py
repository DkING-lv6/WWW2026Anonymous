import torch.nn as nn
import os

from model import Model
from utils import *
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
import random
import dgl
import argparse
from tqdm import tqdm
import time
import numpy as np
import scipy.sparse as sp


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', type=str, default='photo')
parser.add_argument('--lr', type=float)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--embedding_dim', type=int, default=512)
parser.add_argument('--num_epoch', type=int)
parser.add_argument('--auc_test_rounds', type=int, default=256)
parser.add_argument('--negsamp_ratio', type=int, default=1)
parser.add_argument('--mean', type=float, default=0.02)
parser.add_argument('--var', type=float, default=0.01)
args = parser.parse_args()


args.lr = 1e-3

if args.num_epoch is None:
    if args.dataset == 'reddit':
        args.num_epoch = 600
    elif args.dataset == 'tf_finace':
        args.num_epoch = 150
    elif args.dataset == 'elliptic':
        args.num_epoch = 150
    elif args.dataset == 'photo':
        args.num_epoch = 800
    elif args.dataset == 'Amazon':
        args.num_epoch = 800
if args.dataset == 'tf_finace':
    args.weight_decay = 1e-1



print('Dataset:', args.dataset)


dgl.random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


adj, features, labels, all_idx, idx_train, idx_val, idx_test, ano_label, str_ano_label, attr_ano_label, normal_label_idx = load_mat(
    args.dataset)

if args.dataset in ['Amazon', 'reddit']:
    features, _ = preprocess_features(features)
else:
    features = features.todense()

if args.dataset in ['Amazon', 'tf_finace']:
    dgl_graph = adj_to_dgl_graph1(adj)
else:
    dgl_graph = adj_to_dgl_graph(adj)

nb_nodes = features.shape[0]
ft_size = features.shape[1]
raw_adj = adj
adj = normalize_adj(adj)

raw_adj = (raw_adj + sp.eye(raw_adj.shape[0])).todense()
adj = (adj + sp.eye(adj.shape[0])).todense()


features = torch.FloatTensor(features).unsqueeze(0)
adj = torch.FloatTensor(adj).unsqueeze(0)
raw_adj = torch.FloatTensor(raw_adj).unsqueeze(0)
labels = torch.FloatTensor(labels).unsqueeze(0)
normal_label_idx = torch.LongTensor(normal_label_idx)
ano_label = torch.FloatTensor(ano_label)
idx_test = torch.LongTensor(idx_test)


model = Model(ft_size, args.embedding_dim, 'prelu')
optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]))

auc_best = 0
prc_best = 0
best_model_path1 = f'best_model_{args.dataset}.pt'
start = time.time()
'''

'''
with tqdm(total=args.num_epoch) as pbar:
    pbar.set_description('Training')
    for epoch in range(args.num_epoch):
        if epoch % 10 == 0:
            model.eval()
            train_flag = False
            emb, emb_combine, logits, emb_con1, emb_normal, emb_con_fake_abn, emb_con_advanced_abn, loss1 = \
                model(features, adj, idx_test, train_flag, args)

            logits = torch.squeeze(logits[:, idx_test, :]).detach().cpu().numpy()
            test_labels = ano_label[idx_test].cpu().numpy()
            auc = roc_auc_score(test_labels, logits)
            if auc > auc_best:
                auc_best = auc
            ap = average_precision_score(test_labels, logits)
            if ap > prc_best:
                prc_best = ap

end = time.time()

model.load_state_dict(torch.load(best_model_path1))
model.eval()
emb, emb_combine, logits, emb_con1, emb_normal, emb_con_fake_abn, emb_con_advanced_abn, loss1 = \
    model(features, adj, idx_test, False, args)

logits = torch.squeeze(logits[:, idx_test, :]).detach().cpu().numpy()
test_labels = ano_label[idx_test].cpu().numpy()
final_auc = roc_auc_score(test_labels, logits)
final_ap = average_precision_score(test_labels, logits)
print(f"Best AUC after reloading: {final_auc:.4f}")
print(f"Best AP after reloading: {final_ap:.4f}")

print(f"total time：{end - start:.4f} 秒")