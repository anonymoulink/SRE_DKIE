import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from torch.distributions import normal
import numpy as np
# from sinkhorn_knopp import SinkhornKnopp
import torch.nn.functional as F
# from sklearn.metrics import pairwise_distances
import pandas as pd
import random
import warnings
from torch.nn.functional import normalize

warnings.simplefilter(action='ignore', category=Warning)

from requests import Response
from requests import HTTPError
from os import listdir
from transformers import AutoProcessor
from torch.utils.data import Dataset
import torch
from torch.utils.tensorboard import SummaryWriter
import math
from transformers import LayoutLMv3ForTokenClassification
from sklearn.metrics import calinski_harabasz_score

from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
from sklearn.metrics import silhouette_score
# from common import K_Means

random.seed(1)
from torch.utils.data import DataLoader, Sampler
# from t_SNE_cord import K_Means
import time
# from datasets import load_metric
from nltk.corpus import wordnet
import os
import shutil
import os
from collections import ChainMap

os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # 申明环境可以使用的设备
# metric = load_metric("seqeval")
import warnings

warnings.filterwarnings("ignore")
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score)

train = pd.read_pickle('./data/CORD/train.pkl')
val = pd.read_pickle('./data/CORD/val.pkl')
test = pd.read_pickle('./data/CORD/test.pkl')
# 所有的统计
# all_labels = {'menu.nm': 6597, 'menu.price': 2586, 'menu.cnt': 2430, 'total.total_price': 2118,
#               'sub_total.subtotal_price': 1481, 'total.cashprice': 1395, 'total.changeprice': 1299,
#               'sub_total.tax_price': 1279, 'menu.sub_nm': 822, 'menu.unitprice': 750, 'total.menuqty_cnt': 634,
#               'total.creditcardprice': 408, 'menu.discountprice': 403, 'sub_total.service_price': 353,
#               'sub_total.etc': 283, 'sub_total.discount_price': 191, 'menu.sub_cnt': 189, 'menu.sub_price': 160,
#               'total.menutype_cnt': 132, 'total.emoneyprice': 129, 'menu.num': 109, 'total.total_etc': 89,
#               'menu.etc': 19, 'menu.sub_unitprice': 14, 'menu.sub_etc': 9, 'menu.vatyn': 9, 'menu.itemsubtotal': 7,
#               'sub_total.othersvc_price': 6, 'void_menu.nm': 3, 'void_menu.price': 1}
# labels = ['total.menutype_cnt', 'void_menu.nm', 'menu.sub_cnt', 'total.total_etc', 'total.emoneyprice', 'menu.nm',
#           'menu.vatyn', 'menu.sub_price', 'sub_total.service_price', 'menu.cnt', 'menu.sub_unitprice',
#           'total.menuqty_cnt', 'menu.num', 'sub_total.discount_price', 'total.changeprice', 'sub_total.subtotal_price',
#           'menu.price', 'menu.sub_nm', 'void_menu.price', 'sub_total.othersvc_price', 'menu.sub_etc',
#           'menu.itemsubtotal', 'sub_total.etc', 'total.creditcardprice', 'total.cashprice', 'total.total_price',
#           'sub_total.tax_price', 'menu.etc', 'menu.discountprice', 'menu.unitprice']
labels = ['menu.nm', 'menu.price', 'menu.cnt', 'total.total_price', 'sub_total.subtotal_price',
          'total.cashprice', 'total.changeprice', 'sub_total.tax_price', 'menu.unitprice', 'menu.sub_nm',
          'total.menuqty_cnt', 'menu.discountprice', 'total.creditcardprice', 'sub_total.service_price',
          'sub_total.etc', 'sub_total.discount_price', 'menu.sub_cnt', 'menu.sub_price',
          'total.emoneyprice', 'total.menutype_cnt', 'menu.num', 'total.total_etc', 'menu.etc', 'menu.sub_unitprice',
          'menu.sub_etc', 'sub_total.othersvc_price', 'menu.vatyn', 'void_menu.nm', 'void_menu.price',
          'menu.itemsubtotal']
num_labels = len(labels)
print(labels)
label2id = {'menu.nm': 0, 'menu.price': 1,
            'menu.cnt': 2,
            'total.total_price': 3,
            'sub_total.subtotal_price': 4,
            'total.cashprice': 5,
            'total.changeprice': 6,
            'sub_total.tax_price': 7,
            'menu.unitprice': 8,
            'menu.sub_nm': 9,
            'total.menuqty_cnt': 10,
            'menu.discountprice': 11,
            'total.creditcardprice': 12,
            'sub_total.service_price': 13,
            'sub_total.etc': 14,
            'sub_total.discount_price': 15,
            'menu.sub_cnt': 16,
            'menu.sub_price': 17,
            'total.emoneyprice': 18,
            'total.menutype_cnt': 19,
            'menu.num': 20,
            'total.total_etc': 21,
            'menu.etc': 22,
            'menu.sub_unitprice': 23,
            'menu.sub_etc': 24,
            'sub_total.othersvc_price': 25,
            'menu.vatyn': 26,
            'void_menu.nm': 27,
            'void_menu.price': 28,
            'menu.itemsubtotal': 29
            }
# label2id = {'total.menutype_cnt': 0, 'void_menu.nm': 1, 'menu.sub_cnt': 2, 'total.total_etc': 3, 'total.emoneyprice': 4,
#             'menu.nm': 5, 'menu.vatyn': 6, 'menu.sub_price': 7, 'sub_total.service_price': 8, 'menu.cnt': 9,
#             'menu.sub_unitprice': 10, 'total.menuqty_cnt': 11, 'menu.num': 12, 'sub_total.discount_price': 13,
#             'total.changeprice': 14, 'sub_total.subtotal_price': 15, 'menu.price': 16, 'menu.sub_nm': 17,
#             'void_menu.price': 18, 'sub_total.othersvc_price': 19, 'menu.sub_etc': 20, 'menu.itemsubtotal': 21,
#             'sub_total.etc': 22, 'total.creditcardprice': 23, 'total.cashprice': 24, 'total.total_price': 25,
#             'sub_total.tax_price': 26, 'menu.etc': 27, 'menu.discountprice': 28, 'menu.unitprice': 29}
id2label = {
    0: 'menu.nm',
    1: 'menu.price',
    2: 'menu.cnt',
    3: 'total.total_price',
    4: 'sub_total.subtotal_price',
    5: 'total.cashprice',
    6: 'total.changeprice',
    7: 'sub_total.tax_price',
    8: 'menu.unitprice',
    9: 'menu.sub_nm',
    10: 'total.menuqty_cnt',
    11: 'menu.discountprice',
    12: 'total.creditcardprice',
    13: 'sub_total.service_price',
    14: 'sub_total.etc',
    15: 'sub_total.discount_price',
    16: 'menu.sub_cnt',
    17: 'menu.sub_price',
    18: 'total.emoneyprice',
    19: 'total.menutype_cnt',
    20: 'menu.num',
    21: 'total.total_etc',
    22: 'menu.etc',
    23: 'menu.sub_unitprice',
    24: 'menu.sub_etc',
    25: 'sub_total.othersvc_price',
    26: 'menu.vatyn',
    27: 'void_menu.nm',
    28: 'void_menu.price',
    29: 'menu.itemsubtotal'
}


class MemoryBank:
    """ MoCo-style memory bank
    """

    def __init__(self, max_size=1024, embedding_size=128, name='labeled'):
        # assert isinstance(embedding_size, int), f'not implemented for {type(embedding_size)}'
        self.max_size = max_size
        self.embedding_size = embedding_size
        self.name = name
        self.pointer = 0
        self.bank = torch.zeros(self.max_size, self.embedding_size)
        self.label = torch.zeros(self.max_size) - 1  ### -1 denotes invalid
        return

    def add(self, v, y=None):
        assert isinstance(v, torch.Tensor), f'type(v)={type(v)}'
        assert v.shape[1] == self.embedding_size, f'embedding size mismatch, v={v.shape} membank={self.embedding_size}'

        v = v.detach().cpu()
        N = v.size(0)

        idx = (torch.arange(N) + self.pointer).fmod(self.max_size).long()
        self.bank[idx] = v
        if y is not None:
            y = y.cpu()
            self.label[idx] = y.to(self.label.dtype)
        else:
            self.label[idx] = 1
        self.pointer = idx[-1] + 1
        return

    @torch.no_grad()
    def query(self, k=None, device=None):
        if len(self) == 0:
            return None, None
        if k is None:
            idx = (self.label != -1)
            features, labels = self.bank[idx], self.label[idx]
        else:
            if isinstance(k, int):
                idx = (self.label != -1)
                features, labels = self.bank[idx], self.label[idx]
        if device is not None:
            features = features.to(device)
            labels = labels.to(device)
        return features.detach(), labels.detach()

    def debug(self):
        print('membank::DEBUG')
        print(f'name={self.name}, max_size={self.max_size}, embedding_size={self.embedding_size}')
        print(f'pointer={self.pointer}')
        print(f'\tvector={self.bank.shape} {self.bank}')
        print(f'\tlabel={self.label.shape} {self.label}')
        return

    def __len__(self):
        return (self.label != -1).sum().item()


def compute_consensus_gpu(N, K, knn_map):
    """
    compute consensus KNN with range @K on neighborhood @knn_map computed on data
    """
    consensus = torch.zeros(N, N).to(knn_map.device)
    idx = torch.cartesian_prod(torch.arange(K), torch.arange(K)).to(knn_map.device)  # 确保输入的邻接矩阵 A 是 PyTorch 的张量。
    idx = idx[idx[:, 0] != idx[:, 1]]  ### exclude self-self
    for i in range(N):
        consensus[knn_map[i, idx[:, 0]], knn_map[i, idx[:, 1]]] += 1
        consensus[i, knn_map[i, :]] += 1  # 增加共识矩阵中第 i 个数据点的所有邻居的计数
        consensus[knn_map[i, :], i] += 1  # 增加共识矩阵中所有以第 i 个数据点为邻居的数据点的计数。
    return consensus


# 图扩散（graph diffusion）函数
@torch.no_grad()
def graph_diffusion(A, alpha=-1, P=None, step=1):
    """
    TPG diffusion process
    """
    assert isinstance(A, torch.Tensor), 'A should be torch.Tensor'
    Q = A.clone()
    P = A if P is None else P
    for t in range(step):
        Q = (P @ Q) @ P.T + torch.eye(Q.size(0),
                                      device=Q.device)  # P @ Q 和 P.T 是矩阵乘法，torch.eye(Q.size(0), device=Q.device)
        # 创建一个与 Q 同维度的单位矩阵，以确保稳定性
    return Q


@torch.no_grad()
def update_knn_affinity(pos_affinity, mutex_affinity, knn_affinity, neg_affinity):
    """
    update predicted KNN affinity (@knn_affinity and @neg_affinity) with
    must-hold ground-truth affinities (@pos_affinity and @mutex_affinity)
    """
    # print("pos_affinity", pos_affinity.shape)
    # print("knn_affinity", knn_affinity.shape)
    a = (pos_affinity == 1) & (knn_affinity == 0)  ### neglected positives
    b = (mutex_affinity == 1) & (knn_affinity == 1)  ### false positives
    c = (pos_affinity == 1) & (neg_affinity == 1)  ### false negatives
    knn_affinity[a] = 1
    neg_affinity[a] = 0  # mutex
    knn_affinity[b] = 0
    neg_affinity[b] = 1  # mutex
    knn_affinity[c] = 1
    neg_affinity[c] = 0  # mutex
    return knn_affinity, neg_affinity


def compute_cosine_knn(features, k=5):
    """ compute 2d feature KNN query
    Returns:
        knn_ind
        knn_val
    """
    assert len(features.shape) == 2, f'not implemented for shape={features.shape}'
    similarity = torch.mm(features, features.T)
    # 出相似度矩阵中每行的前 k 个最高值
    k_query = similarity.topk(k=k)
    # k_query.indices 获取了这些最高相似度值的索引，即对于矩阵中的每个特征向量，这些索引代表了与其最相似的 k 个特征向量的位置。
    # k_query.values 获取了相应的相似度值，即这些特征向量之间的相似度
    knn_ind = k_query.indices
    knn_val = k_query.values
    return similarity, knn_ind, knn_val


def compute_knn_loss(knn_affinity, neg_affinity, features, features_mb, features_me,
                     temperature=0.07, k=20, epoch=None,
                     ):
    """ compute KNN loss on all features, 2*B x (Q + 1) pairs
    all features assumed to be already normalized, topK neighbors are computed

    Args:
        knn_affinity (Tensor[2*B, M]): affinity map with selected knn `positive` pairs
        neg_affinity (Tensor[2*B, M]): affinity map with selected knn `negative` pairs
        features (Tensor[2*B, D]): features from query encoder
        features_mb (Tensor[M, D]): features from memory bank
        features_me (Tensor[2*B, D]): features from momentum encoder, serving as positive pair for @features
        k (int): number of neighbors
    Returns:
        loss
    """
    assert ((knn_affinity + neg_affinity) == 2).sum() == 0, 'ERROR:: violate the mutex condition'
    device = features.device
    N, D = features.size()
    M = features_mb.size(0)

    ### compute similarity matrix
    pos = torch.cosine_similarity(features, features_me).unsqueeze(1)
    similarity_matrix = torch.mm(features, features_mb.T)
    ### compute label map
    label_map = torch.zeros_like(similarity_matrix).to(device)  ### B x [Mpos, Mneg]
    label_map[knn_affinity == 1] = 1
    label_map[neg_affinity == 1] = -1

    ### concat with positive (different view)
    similarity_matrix = torch.cat([pos, similarity_matrix], dim=1) / temperature
    label_map = torch.cat([torch.ones_like(pos).to(device), label_map], dim=1)
    valid_map = label_map.clone().abs().detach()

    # for numerical stability
    similarity_matrix_copy = similarity_matrix.clone()
    similarity_matrix_copy[valid_map == 0] = -2
    logits_max, _ = torch.max(similarity_matrix_copy, dim=1, keepdim=True)
    logits = similarity_matrix - logits_max.detach()
    # compute log_prob
    exp_logits = torch.exp(logits)
    exp_logits = exp_logits * (valid_map != 0)  ### nonzero for only pos/neg
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    log_prob = log_prob * (valid_map != 0)  ### nonzero for only pos/neg
    label_map = label_map * (label_map != -1)  ### nonzero for only pos
    # compute mean of log-likelihood for each positive pair, then sum for each query sample
    mean_log_prob_pos = (label_map * log_prob).sum(1) / label_map.sum(1)
    loss = - mean_log_prob_pos.mean()
    return loss


@torch.no_grad()
def compute_diffusion_neighbor_consensus_pseudo_knn(features, features_mmb, diffusion_params, k=20,
                                                    transition_prob=None):
    """
    compute SemiAG on the graph generated from student @features and memory bank @features_mb
    SemiAG is controlled by @diffusion_params and neighborhood size @k
    """
    features_mmb = torch.randn(4, 8)

    def compute_consensus_on_features(features, features_mb, k):
        B = features.size(0)
        # 学生的特征向量，形状为 (B, D)，其中 B 是批次大小，D 是特征维度
        print("features", features.shape)  # torch.Size([4, 8])
        print("features_mb", features_mb.shape)  # torch.Size([4, 8])
        stack_features = torch.cat([features, features_mb], dim=0)  ### B+M, D

        K = k
        similarity, knn_ind, _ = compute_cosine_knn(stack_features, k=min(stack_features.size(0), K + 1))
        # similarity index
        knn_map = knn_ind[:, 1:]
        N = knn_map.shape[0]
        # 共识矩阵
        consensus = compute_consensus_gpu(N, K, knn_map)
        return similarity, consensus

    # 这个需要给出的
    diffusion_steps = diffusion_params['diffusion_steps']
    q = diffusion_params['q']

    B = features.size(0)
    # similarity, consensus
    similarity, consensus = compute_consensus_on_features(features, features_mmb, k)
    consensus = consensus / (consensus.sum(1, keepdims=True) + 1e-10)
    # 以上步骤的是共有的&&&&&&&&*******&&&&&*$$$$$$$$$******************
    ### perform graph diffusion # 这是一个图融合策略
    diff_transition = graph_diffusion(consensus, step=diffusion_steps, P=transition_prob)
    ### thresholding
    # 下面就是计算了diff_transition矩阵中非零元素的平均值. diff_transition是一个扩散矩阵，其中的元素表示图中节点间的过渡概率或相似度。
    T = diff_transition[diff_transition.nonzero(as_tuple=True)].mean()
    # 先选取 diff_transition 中大于平均值 T 的元素，然后计算这些元素值的分位数 q。这个分位数用作创建KNN邻接矩阵的阈值。
    q = np.quantile(diff_transition[diff_transition > T].detach().cpu().numpy(), q=q)
    # diff_transition 矩阵中的每个元素是否大于分位数阈值 q，构建一个布尔矩阵 knn_affinity。
    # 这个矩阵可能表示节点之间是否有足够的相似度来被认为是邻居
    knn_affinity = (diff_transition > q)  # 布尔矩阵 knn_affinity
    # similarity 矩阵进行了切片操作，可能是为了获取特定范围内节点的相似度子矩阵。B 可能是一个指定的边界值。
    similarity = similarity[:B, B:]
    knn_affinity = knn_affinity[:B, B:]
    return similarity.detach(), knn_affinity.detach(), consensus[:B, B:].detach()


def compute_cross_contrastive_loss_unsup(q_features, k_features, temperature=0.07, n_views=2):
    """ compute unsup contrastive loss on all features
    assume @q_features and @k_features are already normalized and in the same order, @n_views=2
    """
    device = q_features.device
    B = q_features.size(0) // 2
    q_labels = k_labels = torch.arange(B, device=device).repeat(2)
    labels = (q_labels.view(-1, 1) == k_labels.view(1, -1)).float()  # B x B
    similarity_matrix = torch.mm(q_features, k_features.T)
    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.size(0), dtype=torch.bool).to(device)  # B x B
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    # compute loss
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
    logits = logits / temperature
    unsup_contrastive_loss = torch.nn.CrossEntropyLoss()(logits, labels)
    return unsup_contrastive_loss, logits, labels


@torch.no_grad()
def compute_pos_affinity_with_lmask(class_labels, mask_lab, labels_mb, lmask_mb):
    """
    compute positive/negative affinity (@pos_affinity/@mutex_affinity) that
    must hold from @class_labels and @mask_lab # 当前批次样本的类别标签
    """
    label_match = (class_labels.repeat(2).view(-1, 1) == labels_mb.view(1, -1))  ### 2B x M
    print("label_match", label_match.shape)
    mask_match = (mask_lab.int().repeat(2).view(-1, 1) == 1) * (lmask_mb.int().view(1, -1) == 1)
    pos_affinity = (label_match * mask_match)
    mutex_affinity = (label_match == 0) & (mask_match == 1)
    return pos_affinity, mutex_affinity


@torch.no_grad()
def negative_sampling_from_membank(knn_affinity, similarity, neg_samples=-1):
    """ random sampling on negative samples
    Returns:
        neg_affinity: 1 for neg, 0 for neglected pos
        neg_samples: for `random` method, N neg = @neg_samples - N sim
    """
    if neg_samples == -1:  ### use all
        return (~knn_affinity)

    device = knn_affinity.device
    B, M = knn_affinity.size()

    neg_affinity = torch.zeros_like(knn_affinity).to(device)
    qi, kj = (knn_affinity == 0).nonzero(as_tuple=True)  ### neg ind
    for i in range(B):
        row_select = (qi == i)
        assert row_select.sum() > 0, 'ERROR:: neg not exists'
        idx_select = torch.randperm(row_select.sum().int().item()).to(device)[:neg_samples - knn_affinity[i, :].sum()]
        neg_affinity[qi[row_select][idx_select], kj[row_select][idx_select]] = 1
    return neg_affinity


def annealing_linear_ramup(eta_min, eta_max, t, T=200):
    return eta_min + (eta_max - eta_min) * min(t, T) / T


list1 = [item for item in train[3]]
all_labels = [item for sublist in train[1] for item in sublist]

# category_counts = Counter(all_labels)

# listA是您提供的一个列表
# listA = ['menu.nm', 'menu.price', 'menu.cnt', 'total.total_price', 'sub_total.subtotal_price',
#                    'total.cashprice', 'total.changeprice', 'sub_total.tax_price', 'menu.unitprice', 'menu.sub_nm',
#                    'total.menuqty_cnt', 'menu.discountprice', 'total.creditcardprice', 'sub_total.service_price',
#                    'sub_total.etc', 'sub_total.discount_price', 'menu.sub_cnt', 'menu.sub_price',
#                    'total.emoneyprice', 'total.menutype_cnt', 'menu.num', 'total.total_etc']  # 替换为实际的listA内容
listA = ['menu.etc', 'menu.sub_unitprice',
         'menu.sub_etc', 'sub_total.othersvc_price', 'menu.vatyn', 'void_menu.nm', 'void_menu.price',
         'menu.itemsubtotal']
# the image name of minority class # 14
unlabel_sample = [item3 for item1, item3 in zip(train[1], train[3]) if any(elem in listA for elem in item1)]
# print("list2", len(list2))  # 14
# majority class # all 786
list3 = [item for item in list1 if item not in unlabel_sample]
# print("list3", len(list3))  # 786
# 步骤2: 随机抽取200个元素放入list3
# print("min(int(len(list3) * 0.0325), len(list3))", min(int(len(list3) * 0.0325), len(list3))) # 40
# labeled sample
label_sample = random.sample(list3, min(int(len(list1) * 0.05), len(list3)))  #
# print("label_sample", label_sample)
# unlabeled dataset contains majority class
list5 = [item for item in list3 if item not in label_sample]
# 这个就是我们的unlabeled dataset
unlabel_sample.extend(list5)
# 输出或进一步处理这些数据

# print("unlabeled sample", unlabel_sample)
# print("unlabeled sample_len", len(unlabel_sample))
# print("labeled sample", len(label_sample))

# 统计标签
test_labels = [item for sublist in test[1] for item in sublist]
from collections import Counter
import logging


def pairwise_distances(x, y):
    """
    Compute the pairwise distance matrix between x and y using PyTorch on GPU.
    Args:
    - x (Tensor): A tensor of shape (n_samples_x, n_features)
    - y (Tensor): A tensor of shape (n_samples_y, n_features). If None, use x itself.
    Returns:
    - distances (Tensor): A tensor of shape (n_samples_x, n_samples_y)
    """

    # Compute the distance matrix
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    distances = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    distances = torch.sqrt(torch.clamp(distances, 0.0))
    return distances


out = "results/cord/imbalance/epoch80_distri_align_L4"  # sink改为了3，想加快收敛速度

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S")
logger = logging.getLogger(__name__)

fh = logging.FileHandler(out + '/report.log', "w")

formats = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

fh.setFormatter(formats)

logger.addHandler(fh)

logger.setLevel(logging.DEBUG)


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


# stop words list
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our',
              'ours', 'ourselves', 'you', 'your', 'yours',
              'yourself', 'yourselves', 'he', 'him', 'his',
              'himself', 'she', 'her', 'hers', 'herself',
              'it', 'its', 'itself', 'they', 'them', 'their',
              'theirs', 'themselves', 'what', 'which', 'who',
              'whom', 'this', 'that', 'these', 'those', 'am',
              'is', 'are', 'was', 'were', 'be', 'been', 'being',
              'have', 'has', 'had', 'having', 'do', 'does', 'did',
              'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
              'because', 'as', 'until', 'while', 'of', 'at',
              'by', 'for', 'with', 'about', 'against', 'between',
              'into', 'through', 'during', 'before', 'after',
              'above', 'below', 'to', 'from', 'up', 'down', 'in',
              'out', 'on', 'off', 'over', 'under', 'again',
              'further', 'then', 'once', 'here', 'there', 'when',
              'where', 'why', 'how', 'all', 'any', 'both', 'each',
              'few', 'more', 'most', 'other', 'some', 'such', 'no',
              'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
              'very', 's', 't', 'can', 'will', 'just', 'don',
              'should', 'now', '']


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def synonym_replacement(words, n):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words]))  # 忽略停用词
    # print(random_word_list)
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        # print(synonyms)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            # print("replaced", random_word, "with", synonym)
            num_replaced += 1
        if num_replaced >= n:  # only replace up to n words
            break
    # this is stupid but we need it, trust me
    # sentence = ' '.join(new_words)
    # new_words = sentence.split(' ')
    return new_words


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            if len(synonym.split()) == 1:  # 筛选同义词长度为1
                synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)


def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words


def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words) - 1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words) - 1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words


# list all training image file names
image_files_train = './data/CORD/train/image'  # [f for f in listdir('./data/CORD/train/image')]
# list all test image file names
image_files_dev = './data/CORD/dev/image'  # [f for f in listdir('./data/CORD/dev/image')]
image_files_test = './data/CORD/test/image'  # [f for f in listdir('./data/CORD/test/image')]


# tokenizer = LayoutLMv3TokenizerFast.from_pretrained('microsoft/layoutlmv3-base')
# processor = LayoutLMv3Processor(LayoutLMv3FeatureExtractor(apply_ocr=False), tokenizer)
# encoded_inputs = processor(image, words, boxes=boxes,
#                            return_offsets_mapping=True,
#                            padding="max_length",
#                            return_tensors="pt", truncation=True)
# print(encoded_inputs.keys())
# offset_mapping = encoded_inputs.pop('offset_mapping')
# print(offset_mapping)
# print(np.array(offset_mapping.squeeze().tolist())[:, 0])
#
# device = torch.device('cuda:0')
#
# for k, v in encoded_inputs.items():
#     encoded_inputs[k] = v.to(device)
#
# # load the fine-tuned model from the hub
# model = LayoutLMv3ForTokenClassification.from_pretrained("./results/full_supervised_cord/Checkpoints5")
#
# # id2label = model.config.id2label
# model.to(device)
#
# # forward pass
# outputs = model(**encoded_inputs)
# print(outputs.logits.shape)
#
# predictions = outputs.logits.argmax(-1).squeeze().tolist()
# token_boxes = encoded_inputs.bbox.squeeze().tolist()
# is_subword = np.array(offset_mapping.squeeze().tolist())[:, 0] != 0
#
# true_predictions = [id2label[pred] for idx, pred in enumerate(predictions) if not is_subword[idx]]
# print("true_predictions", (true_predictions)
def complex_inverse_proportion_with_t(a, t):
    return 1 / (np.log(t / (0.5 - a)) + 0.01)


def unnormalize_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]


w_ratio = 0.05
s_ratio = 0.1


def create_batches(data_list, batch_size, tensor=None):
    """
    将列表分成指定大小的批次。
    :param data_list: 要分批的列表。
    :param batch_size: 每个批次的大小。
    :return: 包含分批数据的列表。
    """
    num_batches = math.ceil(len(data_list) / batch_size)  # 总批数,ceil向上取整
    print("num_batches", num_batches)
    batched_data = []
    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, len(data_list))
        # print("end", end)
        if tensor == False:
            batch = torch.from_numpy(np.array(data_list[start:end])).to(device)
            # batch = data_list[start:end]
            # print('len(batch)', len(batch))
        else:
            batch = data_list[start:end]
            # print("batch", batch.shape)
            # batch = data_list[start:end]
        batched_data.append(batch)
        # print("batched_data", batched_data)
    return batched_data


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


class CORDDataset(Dataset):
    """CORD dataset."""

    def __init__(self, annotations, image_dir, image_dir_path, processor=None, max_length=512, train=True):
        """
        Args:
            annotations (List[List]): List of lists containing the word-level annotations (words, labels, boxes).
            image_dir (string): Directory with all the document images.
            processor (LayoutLMv3Processor): Processor to prepare the text + image.
        """
        self.words, self.labels, self.boxes, _ = annotations
        self.image_dir_path = image_dir
        # self.image_file_names = [f for f in listdir(image_dir)]
        self.image_file_names = image_dir_path
        # print("self.image_file_names", self.image_file_names)
        self.processor = processor
        self.train = train

    def __len__(self):
        return len(self.image_file_names)

    def __getitem__(self, idx):
        # first, take an image
        item = self.image_file_names[idx]
        # print("item", item)
        # exit()
        image = Image.open(self.image_dir_path + '/' + item).convert("RGB")

        # get word-level annotations
        words = self.words[idx]
        # print("words", words)
        boxes = self.boxes[idx]
        # print("boxes", boxes)
        word_labels = self.labels[idx]

        assert len(words) == len(boxes) == len(word_labels)  # 18

        word_labels = [label2id[label] for label in word_labels]
        # use processor to prepare everything
        encoding = self.processor(image, words, boxes=boxes, word_labels=word_labels,
                                  padding="max_length", truncation=True,
                                  return_tensors="pt")

        # remove batch dimension
        for k, v in encoding.items():
            encoding[k] = v.squeeze()

        if self.train:

            '''弱增强'''
            words_w = np.array(words)
            indices = np.arange(len(words_w))
            np.random.shuffle(indices)
            samples_index = indices[0: int(len(indices) * w_ratio)]
            samples = words_w[samples_index]
            synonym = synonym_replacement(samples, len(samples))  # 随机取5%的单词进行同义词替换
            words_w[samples_index] = synonym
            words_w = words_w.tolist()
            encoding_w = processor(image, words_w, boxes=boxes, word_labels=word_labels,
                                   truncation=True, padding="max_length", return_tensors="pt")
            '''强增强'''

            words_s = np.array(words)
            indices = np.arange(len(words_s))
            np.random.shuffle(indices)
            samples_index = indices[0: int(len(indices) * s_ratio)]
            samples = words_s[samples_index]
            swap = random_swap(samples, len(samples))  # 随机取10%的单词进行随机交换
            words_s[samples_index] = swap
            words_s = words_s.tolist()
            encoding_s = processor(image, words_s, boxes=boxes, word_labels=word_labels,
                                   truncation=True, padding="max_length", return_tensors="pt")

            for k, v in encoding_w.items():
                encoding_w[k] = v.squeeze()
            for k, v in encoding_s.items():
                encoding_s[k] = v.squeeze()

        assert encoding.input_ids.shape == torch.Size([512])
        assert encoding.attention_mask.shape == torch.Size([512])
        # assert encoding.token_type_ids.shape == torch.Size([512])
        assert encoding.bbox.shape == torch.Size([512, 4])
        assert encoding.pixel_values.shape == torch.Size([3, 224, 224])
        assert encoding.labels.shape == torch.Size([512])

        if self.train:
            # 这里修改了，修改为encoding_w-》encoding
            # return encoding, encoding_w, encoding_s
            return encoding, encoding_s, encoding_s
        else:
            return encoding


def ration(class_counts, epoch):
    over_sampling_ratio = [complex_inverse_proportion_with_t(num / sum(class_counts), epoch) for num in class_counts]
    return over_sampling_ratio


class RFSSampler(Sampler):
    def __init__(self, labels, class_counts, under_sampling_ratio=0.5, over_sampling_ratio=None):
        self.labels = labels
        self.class_counts = class_counts
        self.under_sampling_ratio = under_sampling_ratio
        self.over_sampling_ratio = over_sampling_ratio
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        # 根据self.epoch调整采样策略
        # 例如，每经过10个epochs, 减少欠采样比率:
        self.k = (min(self.class_counts) + max(self.class_counts)) / 2
        self.alphas = [num / sum(self.class_counts) for num in self.class_counts]
        if self.epoch > 10:
            self.over_sampling_ratio = complex_inverse_proportion_with_t(self.alphas, self.epoch)

        mean_samples = sum(self.class_counts) / len(self.class_counts)

        sampled_indices = []

        for idx, count in enumerate(self.class_counts):
            class_indices = [i for i, label in enumerate(self.labels) if label == idx]
            if count > mean_samples:
                # 欠采样
                sampled_indices.extend(random.sample(class_indices, int(sum(count) * self.under_sampling_ratio)))
            else:
                # 过采样
                sampled_indices.extend(class_indices * int(self.over_sampling_ratio))

        random.shuffle(sampled_indices)
        return iter(sampled_indices)

    def __len__(self):
        return len(self.labels)


processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
# train_dataset = CORDDataset(annotations=train,
#                             image_dir=image_files_train,
#                            processor=processor)
image_files_dev_path = [file for file in os.listdir(image_files_dev)]
image_files_test_path = [file for file in os.listdir(image_files_test)]
val_dataset = CORDDataset(annotations=val,
                          image_dir=image_files_dev,
                          image_dir_path=image_files_dev_path,
                          processor=processor)
test_dataset = CORDDataset(annotations=test,
                           image_dir=image_files_test,
                           image_dir_path=image_files_test_path,
                           processor=processor, train=False)
# length = len(train_dataset)
# print("length", length)

split_ratio = 0.2
# # train_lab_size, train_unlab_size = int(split_ratio * length),length-int(split_ratio * length) #有监督：无监督 = 29 ：120
# train_lab_size, train_unlab_size = int(0.1 * length), length - int(0.1 * length)  # 有监督：无监督 = 14 ：135
# # train_lab_size, train_unlab_size = int(0.05*length),length-int(0.05*length) #有监督：无监督 = 7 ：142

# first param is data set to be saperated, the second is list stating how many sets we want it to be.
train_lab_dataset = CORDDataset(annotations=train,
                                image_dir=image_files_train,
                                image_dir_path=label_sample,
                                processor=processor)
train_unlab_dataset = CORDDataset(annotations=train,
                                  image_dir=image_files_train,
                                  image_dir_path=unlabel_sample,
                                  processor=processor)
# train_lab_dataset, train_unlab_dataset = torch.utils.data.random_split(train_dataset,
# [train_lab_size, train_unlab_size])  # 随机划分训练集
'''train'''
# 使用RFS Sampler创建DataLoader

train_lab_dataloader = DataLoader(train_lab_dataset, batch_size=4, shuffle=True)
# val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=4)

# train_unlab_sampler = torch.utils.data.distributed.DistributedSampler(train_unlab_dataset)
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, sampler=train_sampler)

mu = int((1 - split_ratio) / split_ratio)
train_unlab_dataloader_ori = DataLoader(train_unlab_dataset, batch_size=1 * 4, drop_last=True, shuffle=True)

# n_unlabeled_samples = train_unlab_size
warmup = False


class DistAlignEMA(object):
    """
    Distribution Alignment Hook for conducting distribution alignment
    """

    def __init__(self, num_classes, momentum=0.999, p_target_type='uniform', p_target=None):
        super().__init__()
        self.num_classes = num_classes
        self.m = momentum

        # p_target
        self.update_p_target, self.p_target = self.set_p_target(p_target_type, p_target)
        print('distribution alignment p_target:', self.p_target)
        # p_model
        self.p_model = None

    @torch.no_grad()
    def dist_align(self, probs_x_ulb, probs_x_lb=None):
        # update queue
        self.update_p(probs_x_ulb, probs_x_lb)

        # dist align
        probs_x_ulb_aligned = probs_x_ulb * (self.p_target + 1e-6) / (self.p_model + 1e-6)
        probs_x_ulb_aligned = probs_x_ulb_aligned / probs_x_ulb_aligned.sum(dim=-1, keepdim=True)
        return probs_x_ulb_aligned

    @torch.no_grad()
    def update_p(self, probs_x_ulb, probs_x_lb):
        # check device
        if not self.p_target.is_cuda:
            self.p_target = self.p_target.to(probs_x_ulb.device)

        probs_x_ulb = probs_x_ulb.detach()
        if self.p_model == None:
            self.p_model = torch.mean(probs_x_ulb, dim=0)
        else:
            self.p_model = self.p_model * self.m + torch.mean(probs_x_ulb, dim=0) * (1 - self.m)

        if self.update_p_target:
            assert probs_x_lb is not None
            self.p_target = self.p_target * self.m + torch.mean(probs_x_lb, dim=0) * (1 - self.m)

    def set_p_target(self, p_target_type='uniform', p_target=None):
        assert p_target_type in ['uniform', 'gt', 'model']

        # p_target
        update_p_target = False
        if p_target_type == 'uniform':
            p_target = torch.ones((self.num_classes,)) / self.num_classes
        elif p_target_type == 'model':
            p_target = torch.ones((self.num_classes,)) / self.num_classes
            update_p_target = True
        else:
            assert p_target is not None
            if isinstance(p_target, np.ndarray):
                p_target = torch.from_numpy(p_target)

        return update_p_target, p_target


def normalize_d(x):
    x_sum = torch.sum(x, dim=-1, keepdim=True)
    x = x / x_sum
    return x.detach()


def compute_class_centers(features, labels):
    unique_labels = np.unique(labels)
    centers = []
    # 字典
    # centers = {}
    for lbl in unique_labels:
        # centers[lbl] = np.mean(features[labels == lbl], axis=0)
        centers.extend(np.mean(features[labels == lbl], axis=0))
    return centers


def distribution(features, labels):
    # labels = torch.from_numpy(labels)
    # features = torch.from_numpy(features)
    # print("labels", labels)
    # print("num_labels_len", torch.unique(labels))
    # print("features.shape[0]", features.shape[0])
    unique_labels = torch.unique(labels)
    unique_labels = unique_labels[unique_labels != -100]  # 移除无效标签
    prototypes = torch.zeros(len(unique_labels), features.shape[1])  # 7, 128
    for i, label in enumerate(unique_labels):
        # select features by label
        # labels = torch.from_numpy(labels)
        # class_mask = (labels == label)
        class_samples = features[labels == label]
        if class_samples.nelement() == 0:
            continue
        # 计算当前类别的原型，即特征的平均值
        prototypes[i] = class_samples.mean(dim=0)
        # cls_inds = torch.where(labels == idx)[0]    # 统计labels中各类的位置,即token所在的位置,labels应该是要展开的，返回的是一个tuple,所以要取[0],2048
        # if len(cls_inds):
        #     feats_selected = features[cls_inds] # features[2048, 128]，在2048个token钟选择单个类别的feat
        #     prototypes[idx] = feats_selected.mean(0)
    print('prototypes', prototypes.shape)  # predicted_class, 128
    return prototypes


class FreeMatchThreshold(object):
    """
    SAT in FreeMatch
    """

    def __init__(self, num_classes, momentum=0.999, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.m = momentum

        self.p_model = torch.ones((self.num_classes)) / self.num_classes  # 初始化为1/C
        self.p_model_ulb = torch.ones((self.num_classes)) / self.num_classes  # 初始化为1/C
        self.p_model_ulb_s = torch.ones((self.num_classes)) / self.num_classes  # 初始化为1/C
        self.label_hist = torch.ones((self.num_classes)) / self.num_classes  # 初始化为1/C
        # self.time_p = None

    @torch.no_grad()
    def update(self, use_quantile=False, clip_thresh=False, probs_x_lb=None, probs_x_ulb=None, probs_x_ulb_s=None):
        # 更新label_hist：
        # p_model: 模型对每个类c预测的期望值
        # time_p：全局阈值

        max_probs, max_idx = torch.max(probs_x_ulb, dim=-1, keepdim=True)  # 输出仍然是二维的

        # 未经过softmax
        if not self.p_model.is_cuda:
            self.p_model = self.p_model.to(probs_x_ulb.device)
        if not self.p_model_ulb.is_cuda:
            self.p_model_ulb = self.p_model_ulb.to(probs_x_ulb.device)
        if not self.p_model_ulb_s.is_cuda:
            self.p_model_ulb_s = self.p_model_ulb_s.to(probs_x_ulb.device)
        if not self.label_hist.is_cuda:
            self.label_hist = self.label_hist.to(probs_x_ulb.device)

        self.p_model = self.p_model * self.m + (1 - self.m) * probs_x_lb.view(-1, num_labels).mean(dim=0)
        self.p_model_ulb = self.p_model_ulb * self.m + (1 - self.m) * probs_x_ulb.view(-1, num_labels).mean(dim=0)
        hist = torch.bincount(max_idx.reshape(-1), minlength=self.p_model_ulb.shape[0]).to(self.p_model_ulb.dtype)
        # # 统计不同类别样本的数量
        self.label_hist = self.label_hist * self.m + (1 - self.m) * (hist / hist.sum())
        # print("p_model", self.p_model)
        # print("p_model_ulb", self.p_model_ulb)
        # print("p_model_ulb_s", self.p_model_ulb_s)
        # print("label_hist", self.label_hist)

    def masking(self, probs_x_lb=None, probs_x_ulb=None, probs_x_ulb_s=None, threshold=0.95):

        # max_probs, max_idx = torch.max(probs_x_ulb, dim=-1, keepdim=True)  # 输出仍然是三维的

        with torch.no_grad():
            distri_ = torch.ones_like(self.p_model) - self.p_model
            # distri_ = 1 / self.p_model
            _T_ = 1.0
            distri_ = normalize_d(distri_ / _T_)  # 对应论文(12)Norm(1-q)
            origin = (torch.ones((self.num_classes)) / self.num_classes).to(device)  # 初始化为1/C

            factor = distri_ / origin
            # print("distri", self.p_model)
            # print("distri_", distri_)
            # y_pred_sup = max_probs_mean * probs_x_lb + (1 - max_probs_mean) * probs_x_lb * (distri_ / self.p_model)

            # softmax_y_pred_unsup_w = max_probs_mean * probs_x_ulb * (self.p_model / self.p_model_ulb) + (1 - max_probs_mean) * probs_x_ulb * (distri_ / self.p_model_ulb)
            probs_x_ulb_DA = probs_x_ulb * (self.p_model / self.p_model_ulb)  # [2048,7]
            # probs_x_ulb_reverse = probs_x_ulb_DA * (distri_/distri_.min())
            probs_x_ulb_reverse = probs_x_ulb_DA * factor

            probs_x_ulb_s_DA = probs_x_ulb * (self.p_model / self.p_model_ulb_s)  # [2048,7]

            probs_x_ulb_s_reverse = normalize_d(probs_x_ulb_s_DA * factor)

        # probs_x_ulb_fuse = self.label_hist/(self.label_hist).max() * probs_x_ulb_DA + (1. - self.label_hist/(self.label_hist).max()) * probs_x_ulb_reverse

        # probs_x_ulb_reverse = probs_x_ulb_DA * distri_

        # softmax_y_pred_unsup_w = probs_x_ulb * (self.p_model / self.p_model_ulb)
        # print("probs_x_ulb_reverse",probs_x_ulb_reverse)

        max_probs, target_u = torch.max(probs_x_ulb_reverse, dim=-1)  # values,indices

        mask = max_probs.ge(threshold).to(max_probs.dtype)
        # mask = 0
        pseudo = F.one_hot(target_u, num_classes=num_labels)

        return pseudo, mask, probs_x_ulb_reverse, probs_x_ulb_s_reverse, target_u, factor, distri_


from collections import defaultdict


class FeatureQueue(object):
    def __init__(self, classwise_max_size=None, temp_class=None, initial_prototypes=None, bal_queue=False):
        self.num_classes = temp_class.shape[0]
        self.feat_dim = 128
        self.max_size = 256

        self._bank = defaultdict(lambda: torch.empty(0, self.feat_dim).to(device))
        # self.prototypes = torch.zeros(self.num_classes, self.feat_dim).to(device)  # 7, 128
        self.prototypes = initial_prototypes.to(device)  # 30, 128

        self.classwise_max_size = classwise_max_size
        self.bal_queue = bal_queue

    def enqueue(self, features: torch.Tensor, labels: torch.Tensor):
        for idx in range(self.num_classes):
            # per class max siz
            max_size = (
                    self.classwise_max_size[idx] * 5  # 5x samples
            ) if self.classwise_max_size is not None else self.max_size
            if self.bal_queue:
                max_size = self.max_size
            # select features by label
            cls_inds = torch.where(labels == idx)[
                0]  # 统计labels中各类的位置,即token所在的位置,labels应该是要展开的，返回的是一个tuple,所以要取[0],2048
            if len(cls_inds):
                with torch.no_grad():
                    # push to the memory bank
                    feats_selected = features[cls_inds]  # features[2048, 128]，在2048个token钟选择单个类别的feat
                    self._bank[idx] = torch.cat([self._bank[idx], feats_selected], 0)

                    # fixed size
                    current_size = len(self._bank[idx])
                    if current_size > max_size:
                        self._bank[idx] = self._bank[idx][current_size - max_size:]

                    # update prototypes
                    self.prototypes[idx, :] = self._bank[idx].mean(0)


class FeatureQueue_s(object):

    def __init__(self, classwise_max_size=None, bal_queue=False, num_labels=30):
        self.num_classes = num_labels
        self.feat_dim = 128
        self.max_size = 256

        self._bank = defaultdict(lambda: torch.empty(0, self.feat_dim).to(device))
        self.prototypes = torch.zeros(self.num_classes, self.feat_dim).to(device)  # 7, 128

        self.classwise_max_size = classwise_max_size
        self.bal_queue = bal_queue

    def enqueue(self, features: torch.Tensor, labels: torch.Tensor):
        for idx in range(self.num_classes):
            # per class max size
            max_size = (
                    self.classwise_max_size[idx] * 5  # 5x samples
            ) if self.classwise_max_size is not None else self.max_size
            if self.bal_queue:
                max_size = self.max_size
            # select features by label
            cls_inds = torch.where(labels == idx)[
                0]  # 统计labels中各类的位置,即token所在的位置,labels应该是要展开的，返回的是一个tuple,所以要取[0],2048
            if len(cls_inds):
                with torch.no_grad():
                    # push to the memory bank
                    feats_selected = features[cls_inds]  # features[2048, 128]，在2048个token钟选择单个类别的feat
                    self._bank[idx] = torch.cat([self._bank[idx], feats_selected], 0)

                    # fixed size
                    current_size = len(self._bank[idx])
                    if current_size > max_size:
                        self._bank[idx] = self._bank[idx][current_size - max_size:]

                    # update prototypes
                    self.prototypes[idx, :] = self._bank[idx].mean(0)


from copy import deepcopy


class ModelEMA(object):
    def __init__(self, device, model):
        self.ema = deepcopy(model)
        self.ema.to(device)
        self.ema.eval()
        self.ema_has_module = hasattr(self.ema, 'module')
        # Fix EMA. https://github.com/valencebond/FixMatch_pytorch thank you!
        self.param_keys = [k for k, _ in self.ema.named_parameters()]
        self.buffer_keys = [k for k, _ in self.ema.named_buffers()]
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model, decay):
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()
            for k in self.param_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                model_v = msd[j].detach()
                ema_v = esd[k]
                esd[k].copy_(ema_v * decay + (1. - decay) * model_v)

            for k in self.buffer_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                # print("ema", esd.keys())  # 应包括键 'layoutlmv3.embeddings.position_ids'
                # print("model", msd.keys())  # 应包括对应于 'j' 的键
                if 'layoutlmv3.embeddings.position_ids' in msd:
                    esd['layoutlmv3.embeddings.position_ids'].copy_(msd['layoutlmv3.embeddings.position_ids'])
                else:
                    print("源字典中不存在键 'layoutlmv3.embeddings.position_ids'.")
                esd[k].copy_(msd[j])


# model = MeanTeacherNet()
# model = LayoutLMv3ForTokenClassification.from_pretrained('microsoft/layoutlmv3-base',
#                                                          num_labels=num_labels)
model = LayoutLMv3ForTokenClassification.from_pretrained('microsoft/layoutlmv3-base',
                                                         num_labels=num_labels)
# label_num_labels = 22
#
# # 仅替换分类头中的输出投影层
# model.classifier.out_proj = torch.nn.Linear(768, label_num_labels)
# # print("model_keeeee", model)
# exit()epoch80_distri_align_L4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cuda:2')
model.to(device)
checkpoint = torch.load('./results/cord/imbalance/epoch80_distri_align_L4_33/model_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'], strict=False)

ema_model = ModelEMA(device, model)
# model = torch.nn.DataParallel(model.cuda(), device_ids = [0,1,2,3])
# optimizer = AdamW(model.parameters(), lr=5e-5)  # 教师模型参数更新 # 学生模型主要用于验证和测试
optimizer = AdamW(model.parameters(), lr=1e-5)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.03,
# momentum=0.9,
# weight_decay=5e-4)

from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=0.5,
                                    last_epoch=-1):
    # Warmup预热学习率：先从一个较小的学习率线性增加至原来设置的学习率，再进行学习率的cos衰减
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / \
                   float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def get_polynomial_decay_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps, lr_end=1e-7, power=1.0, last_epoch=-1
):
    """
    Create a schedule with a learning rate that decreases as a polynomial decay from the initial lr set in the
    optimizer to end lr defined by `lr_end`, after a warmup period during which it increases linearly from 0 to the
    initial lr set in the optimizer.
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        lr_end (:obj:`float`, `optional`, defaults to 1e-7):
            The end LR.
        power (:obj:`float`, `optional`, defaults to 1.0):
            Power factor.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
    Note: `power` defaults to 1.0 as in the fairseq implementation, which in turn is based on the original BERT
    implementation at
    https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/optimization.py#L37
    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_init = optimizer.defaults["lr"]
    assert lr_init > lr_end, f"lr_end ({lr_end}) must be be smaller than initial lr ({lr_init})"

    def lr_lambda(current_step: int):
        current_step = current_step + 1
        if current_step < num_warmup_steps:
            return max(float(current_step) / float(max(1, num_warmup_steps)), 1e-2)
        elif current_step > num_training_steps:
            return lr_end / lr_init  # as LambdaLR multiplies by lr_init
        else:
            lr_range = lr_init - lr_end
            decay_steps = num_training_steps - num_warmup_steps
            pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
            decay = lr_range * pct_remaining ** power + lr_end
            return decay / lr_init  # as LambdaLR multiplies by lr_init

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# global_step = 0
# num_train_epochs = math.ceil(total_steps / eval_steps)    #这个是预先设的total_steps和eval_steps
num_train_epochs = 80

alpha = 0.1
threshold = 0.95
T = 1.0

# thresholding module with convex mapping function x / (2 - x)
# thresholding_module = DynamicThresholdingModule(threshold, True,
#                                                 lambda x: (torch.log(x + 1.) + 0.5) / (math.log(2) + 0.5), num_labels,
#                                                 train_unlab_size, device)
Dynamic_thres = FreeMatchThreshold(num_classes=num_labels, momentum=0.999)
DistAlign = DistAlignEMA(num_classes=num_labels)

epsilon = 0.05  # 0.05改为0.5
sinkhorn_iterations = 3  # 这里改为了3


def distributed_sinkhorn(out):
    Q = torch.exp(out / epsilon).t()  # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1]  # number of samples to assign
    K = Q.shape[0]  # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q /= sum_Q

    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B  # the colomns must sum to 1 so that Q is an assignment
    return Q.t()


class WeightedF1Loss(torch.nn.Module):
    def __init__(self, class_weights):
        super(WeightedF1Loss, self).__init__()
        self.class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    def forward(self, probs, targets):
        # 转换为对数概率
        num_labels = 30
        # log_probs = torch.log(probs + 1e-6)
        # 计算交叉熵损失
        # print("probs", probs.view(-1, num_labels).shape)
        # print("targets.view(-1)", targets.view(-1).shape)
        # loss_cross = torch.nn.CrossEntropyLoss(weight=self.class_weights, reduction='none')
        # l1_loss = F.l1_loss(probs.view(-1, num_labels), targets.view(-1), reduction='none')
        # weighted_l1_loss = l1_loss * self.class_weights
        # weighted_l1_loss = l1_loss * self.class_weights
        weighted_l1_loss = F.cross_entropy(probs[:, 0, :], targets, weight=weights)

        return weighted_l1_loss.mean()


def compute_distance_matrix(A, B, device):
    def euclidean_distance_manual(x, y):
        squared_diff_sum = 0.0
        for xi, yi in zip(x, y):
            squared_diff_sum += (xi - yi) ** 2
        return math.sqrt(squared_diff_sum.item())

    distance_matrix = torch.zeros((len(A), len(B))).to(device)
    # distance_matrix = {}
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            distance_matrix[i][j] = euclidean_distance_manual(a, b)
    return distance_matrix


def embedding_distance_matrix(eddding, cluster_centers_unlabel, prototypes):
    """
    Compute the Euclidean distance matrix between two lists of tensors A and B.
    prototypes: list
    Parameters:
    - A: List of tensors representing center cluster A.
    - B: List of tensors representing center cluster B.

    Returns:
    - Distance matrix where element (i, j) is the distance between A[i] and B[j].
    """
    data_points_expanded = cluster_centers_unlabel.unsqueeze(1)  # 变为形状 (30, 1, 128)
    prototype_expanded = prototypes.unsqueeze(0)  # 变为形状 (1, 7, 128)
    # print("prototype_expanded", prototype_expanded.shape)
    # print("data_points_expanded", data_points_expanded.shape)
    # 现在我们可以相减并广播到形状 (30, 7, 128)
    differences = data_points_expanded - prototype_expanded
    # 计算平方和后的欧式距离
    distances = torch.sqrt(torch.sum(differences ** 2, dim=2))
    # print("distances", distances.shape)
    return distances


def results_test(preds, out_label_ids):
    preds = np.argmax(preds, axis=2)

    # label_map = {i: label for i, label in enumerate(labels)}
    # label_map = {0: 'O', 1: 'B-HEADER', 2: 'I-HEADER', 3: 'B-QUESTION', 4: 'I-QUESTION', 5: 'B-ANSWER', 6: 'I-ANSWER'}
    label_map = id2label

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    out_label_list_ = []
    preds_list_ = []

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != -100:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

                out_label_list_.append(out_label_ids[i][j])
                preds_list_.append(preds[i][j])

    results = {
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
        "accuracy": accuracy_score(out_label_list, preds_list)
    }

    labelsss = ['total.menutype_cnt', 'void_menu.nm', 'menu.sub_cnt', 'total.total_etc', 'total.emoneyprice', 'menu.nm',
                'menu.vatyn', 'menu.sub_price', 'sub_total.service_price', 'menu.cnt', 'menu.sub_unitprice',
                'total.menuqty_cnt', 'menu.num', 'sub_total.discount_price', 'total.changeprice',
                'sub_total.subtotal_price',
                'menu.price', 'menu.sub_nm', 'void_menu.price', 'sub_total.othersvc_price', 'menu.sub_etc',
                'menu.itemsubtotal', 'sub_total.etc', 'total.creditcardprice', 'total.cashprice', 'total.total_price',
                'sub_total.tax_price', 'menu.etc', 'menu.discountprice', 'menu.unitprice']
    labelss = np.arange(len(labelsss)).tolist()
    cm_A = confusion_matrix(out_label_list_, preds_list_, labels=labelss)
    # print("cm_A", cm_A)
    cm_A_norm = (cm_A / cm_A.sum(axis=1)[:, np.newaxis])  # 相当于keepdim = True

    fig, axs = plt.subplots(figsize=(10, 10))

    sns.heatmap(cm_A_norm, annot=True, cmap='Blues', center=None, fmt='.2f', linewidths=0.5, square=True,
                xticklabels=labelsss, yticklabels=labelsss, ax=axs, annot_kws={'size': 5}, vmin=0, vmax=1)
    axs.set_title("Imbalance", fontsize=20)
    axs.set_xlabel('Predicted label', fontsize=20)
    axs.set_ylabel('True label', fontsize=20)
    plt.tight_layout()

    is_best = results['f1'] > best_f1  ########改为只保存最好的结果
    if is_best:
        plt.savefig(out + '/confusion_matrix.png', format='png', dpi=300)
    # 画出某个图的token位置图
    return results, classification_report(out_label_list, preds_list)


from collections import Counter
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE
import sklearn.datasets as datasets
# import cudf
# from cuml.cluster import DBSCAN
from scipy.spatial import distance_matrix
from itertools import chain


def plot_TSNE(embeddings, y, path):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_tsne = tsne.fit_transform(embeddings)

    colors = ['#57f644', '#01c26d', '#ae899b', '#c3667e', '#c6498e', '#0a9a8e', '#4a0fe5', '#dd4415', '#9fa7d8',
              '#86860a',
              '#394dbf', '#11d271', '#ba8a8b', '#fd0a59', '#3722e9', '#1f95e9', '#003ae6', '#99a182', '#35b984',
              '#7af380',
              '#45fe05', '#f6ffd9', '#2b5ce8', '#20369b', '#f6e6e5', '#461342', '#169e0b', '#b44073', '#e4cb0a',
              '#7f8c68']  #######颜色记得改，有时候不是30个类别
    # labels_ = ['total.menutype_cnt', 'void_menu.nm', 'menu.sub_cnt', 'total.total_etc', 'total.emoneyprice', 'menu.nm',
    #       'menu.vatyn', 'menu.sub_price', 'sub_total.service_price', 'menu.cnt', 'menu.sub_unitprice',
    #       'total.menuqty_cnt', 'menu.num', 'sub_total.discount_price', 'total.changeprice', 'sub_total.subtotal_price',
    #       'menu.price', 'menu.sub_nm', 'void_menu.price', 'sub_total.othersvc_price', 'menu.sub_etc',
    #       'menu.itemsubtotal', 'sub_total.etc', 'total.creditcardprice', 'total.cashprice', 'total.total_price',
    #       'sub_total.tax_price', 'menu.etc', 'menu.discountprice', 'menu.unitprice']  #类别记得改
    labels_ = [str(i) for i in range(len(id2label))]
    # Plot the t-SNE visualization
    plt.figure(figsize=(8, 6))

    for i in range(len(labels_)):
        plt.scatter(embeddings_tsne[y == i, 0], embeddings_tsne[y == i, 1], color=colors[i], label=labels_[i])

    plt.legend(bbox_to_anchor=(0.5, 0.97, 0.0, 0.0), loc='lower center', ncol=8, frameon=False, mode='wrap')
    plt.show()
    plt.savefig(path)


best_f1 = 0
rel_precision = 0
rel_recall = 0
rel_accuracy = 0

similarity_fn = torch.nn.CosineSimilarity(dim=2)

empt = torch.zeros(128).to(device)  # 128

from sklearn.mixture import GaussianMixture
import random
from torch.utils.data import DataLoader, Sampler

CE = torch.nn.CrossEntropyLoss(reduction='none')
CEloss = torch.nn.CrossEntropyLoss()
import torch.nn as nn
import torch.nn.functional as F


def fit_gmm_to_distances(distances, gmm):
    gmm_params = {}
    for k, dk in enumerate(distances):
        # 转换为numpy，因为sklearn不接受torch tensor
        # dk_np = dk.cpu().detach().numpy().reshape(-1, 1)
        dk_np = dk.reshape(-1, 1)
        # gmm = GaussianMixture(n_components=2, covariance_type='full')
        gmm.fit(dk_np)
        gmm_params[k] = gmm
    return gmm_params


def imbalanced_loss(representations, prototypes, labels, class_weights):
    """
    Args:
        representations (Tensor): The representations of the samples.
        prototypes (Tensor): The prototypes of each class.
        labels (Tensor): The labels of the samples.
        class_weights (Tensor): The weight for each class, to handle class imbalance.
    Returns:
        Tensor: The calculated loss.
    """
    cores_w = torch.mm(representations, prototypes.t())

    loss = 0.0
    for i in range(len(representations)):
        rep = representations[i]
        label = labels[i]
        correct_proto = prototypes[label]
        incorrect_protos = torch.cat([prototypes[:label], prototypes[label + 1:]])

        # Distance to the correct prototype
        correct_dist = F.mse_loss(rep, correct_proto)

        # Distance to the incorrect prototypes
        incorrect_dist = torch.min(F.mse_loss(rep.unsqueeze(0), incorrect_protos, reduction='none'))

        # Weighted loss
        loss += class_weights[label] * (correct_dist + torch.max(torch.tensor(0.0), 1 - incorrect_dist))

    return loss / len(representations)


def classify_data(gmm_params, un_outputs, distances, labels):
    labels = labels.tolist()
    # print("labels_len", len(labels))
    align_data = {}
    unalign_data = {}
    align_label = []
    unalign_label = []
    for k, gmm in gmm_params.items():
        # 检索与当前标签 k 对应的距离
        dk = distances[k]
        # 使用 GMM 得到每个样本属于各个组件的概率
        proba = gmm.predict_proba(dk.reshape(-1, 1))
        # 基于概率判断每个样本是否“对齐”
        clean_idx = proba[:, 0] > proba[:, 1]
        noise_idx = ~clean_idx  # 假设第二个分量是“噪声”的数据

        # 对于每个标签 k，找出所有对齐和不对齐的样本索引
        current_clean_indices = np.where((clean_idx) & (labels == k))[0]
        current_noise_indices = np.where((noise_idx) & (labels != k))[0]
        # 将索引添加到列表中
        align_label.extend(current_clean_indices.tolist())
        unalign_label.extend(current_noise_indices.tolist())
        # 选择出与标签 k 对齐和不对齐的输出
        align_data[k] = un_outputs[current_clean_indices]
        unalign_data[k] = un_outputs[current_noise_indices]
    # print("align_data", align_data)
    # print("unalign_data", unalign_data)
    return align_data, unalign_data, align_label, unalign_label


def compute_pseudo_knn(features, features_mb, diffusion_params, k=20, transition_prob=None):
    """
    compute SemiAG on the graph generated from student @features and memory bank @features_mb
    SemiAG is controlled by @diffusion_params and neighborhood size @k
    """

    def compute_consensus_on_features(features, features_mb, k):
        B = features.size(0)
        stack_features = torch.cat([features, features_mb], dim=0)  ### B+M, D
        K = k
        similarity, knn_ind, _ = compute_cosine_knn(stack_features, k=min(stack_features.size(0), K + 1))
        knn_map = knn_ind[:, 1:]
        N = knn_map.shape[0]
        consensus = compute_consensus_gpu(N, K, knn_map)
        return similarity, consensus

    diffusion_steps = diffusion_params['diffusion_steps']
    q = diffusion_params['q']

    B = features.size(0)
    similarity, consensus = compute_consensus_on_features(features, features_mb, k)
    consensus = consensus / (consensus.sum(1, keepdims=True) + 1e-10)
    ### perform graph diffusion
    diff_transition = graph_diffusion(consensus, step=diffusion_steps, P=transition_prob)
    ### thresholding
    T = diff_transition[diff_transition.nonzero(as_tuple=True)].mean()
    q = np.quantile(diff_transition[diff_transition > T].detach().cpu().numpy(), q=q)
    knn_affinity = (diff_transition > q)
    ### compute on samples from student
    similarity = similarity[:B, B:]
    knn_affinity = knn_affinity[:B, B:]
    return similarity.detach(), knn_affinity.detach(), consensus[:B, B:].detach()


def loss_func(reshaped_softmax_y_pred_unsup_s, pseudo, cls_weights):
   # 根据类别统计计算出的权重
   # print("mask", mask)
   # 计算 L1 损失
   l1_loss = F.l1_loss(reshaped_softmax_y_pred_unsup_s, pseudo, reduction='none')

   # 应用类别权重
   weighted_loss = l1_loss * cls_weights
   return weighted_loss
def loss_func_mask(reshaped_softmax_y_pred_unsup_s, pseudo, cls_weights, mask):
   # 根据类别统计计算出的权重
   # print("mask", mask)
   # print("mask_type", type(mask))
   valid_softmax_y_pred_unsup_s = reshaped_softmax_y_pred_unsup_s[mask.bool()]
   valid_pseudo = pseudo[mask.bool()]

   # 计算 L1 损失
   l1_loss = F.l1_loss(valid_softmax_y_pred_unsup_s, valid_pseudo, reduction='none')

   # 应用类别权重
   weighted_loss = l1_loss * cls_weights
   return weighted_loss

def count_elements(lst):
    count_dict = {}
    print("lst", lst)
    for element in lst:
        if element in count_dict:
            count_dict[element] += 1
        else:
            count_dict[element] = 1
    return count_dict


def flatten(lst):
    return [item for sublist in lst for item in sublist]


class PredictionsQueue:
    def __init__(self, max_length, queue):
        """
        初始化一个用于存储预测值的队列。

        :param max_length: 队列的最大长度 L。
        """
        self.max_length = max_length
        self.queue = queue

    def update(self, new_predictions):
        """
        更新队列，添加新的预测值，并保持队列长度不超过最大长度。
        :param new_predictions: 一个列表，包含新的预测值。
        """
        # if isinstance(new_predictions, np.ndarray):
        #     new_predictions = new_predictions.tolist()
        # 将新预测值添加到队列中
        # for pred in new_predictions:
        self.queue.append(new_predictions)
        # self.queue.extend(new_predictions)
        # self.queue.append(new_predictions)
        # print("self.queue", self.queue)

        # 如果队列长度超过最大长度，则移除最早的数据
        if len(self.queue) > self.max_length:
            # 移除最早的数据，以保持队列长度为 max_length
            self.queue = self.queue[-self.max_length:]

    def __len__(self):
        """
        返回队列的当前长度。
        """
        return len(self.queue)

    def __iter__(self):
        """
        Return an iterator over the elements of the queue.
        """
        return iter(self.queue)


import torch


class SimHashTorch:
    def __init__(self, num_bits):
        self.num_bits = num_bits
        self.random_vectors = torch.randn((num_bits), device=device)
        # self.random_vectors =

    def compute_hash(self, vector):
        # print("vector", vector.shape)
        if vector.ndim != 1 or vector.shape[0] != self.num_bits:
            raise ValueError("Input vector must be 1-dimensional and of size equal to 'num_bits'")
        bools = torch.matmul(vector, self.random_vectors) > 0
        # print("bools", bools)
        # Convert 'bools' to a 1-dimensional tensor if it is a 0-d tensor (scalar)
        if bools.ndim == 0:
            bools = bools.unsqueeze(0)
        return ''.join('1' if b.item() else '0' for b in bools)

    def hamming_distance(self, hash1, hash2):
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))


def update_knn_affinity_with_simhash_torch(pos_affinity, knn_affinity, neg_affinity, data_vectors, simhash):
    for index, vector in enumerate(data_vectors):
        current_hash = simhash.compute_hash(vector)
        candidates = []
        for other_index, other_vector in enumerate(data_vectors):
            if other_index != index:
                other_hash = simhash.compute_hash(other_vector)
                if simhash.hamming_distance(current_hash, other_hash) <= 0.9:  # 设定汉明距离的阈值
                    candidates.append(other_index)

        # 根据候选项更新亲和力
        for candidate in candidates:
            if pos_affinity[index, candidate] == 1:
                knn_affinity[index, candidate] = 1
                neg_affinity[index, candidate] = 0
            else:
                knn_affinity[index, candidate] = 0
                neg_affinity[index, candidate] = 1

    return knn_affinity, neg_affinity


# dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
def train(train_lab_dataloader, train_unlab_dataloader_ori, train_unlab_dataset, test_dataloader,
          model, optimizer, ema_model):  # 在这里把scheduler去掉了

    # put the model in training mode
    model.train()

    global best_f1, rel_precision, rel_recall, rel_accuracy
    test_f1s = []
    end = time.time()
    warmup_epochs = 40

    # # ## MOCO initial
    # sup_con_crit = SupConLossWithMembank()

    # class_label = nn.Linear(num_labels, 22).to(device)
    loss_fct = torch.nn.CrossEntropyLoss()
    # 这里出事
    unoutputs_list = []
    predicted_list = []
    targets_list = []
    outputs_list = []
    membank_size = 640
    feat_dim = 128
    best_calinski_harabasz = -np.inf
    # best_score = -1
    best_labels = None

    membank_lmask = MemoryBank(max_size=membank_size, embedding_size=1, name='unsup_lmask')  ### for labeled-or-not
    membank_label = MemoryBank(max_size=2048, embedding_size=1, name='unsup_lmask')  ### for labeled-or-not
    membank_unsup_z = MemoryBank(max_size=membank_size, embedding_size=feat_dim, name='unsup_z')  ### for CLS embedding
    membank_unsup_dpr_z = MemoryBank(max_size=membank_size, embedding_size=feat_dim,
                                     name='unsup_dpr_z')  ### for DPR embeddin

    max_iteraion = (num_train_epochs - 19) * len(train_unlab_dataloader_ori)

    for epoch in range(40, num_train_epochs):

        combined_list = {}
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        losses_imbalance = AverageMeter()
        losses_cls = AverageMeter()
        # losses_u_contr = AverageMeter()
        losses_d2d = AverageMeter()
        test_losses = AverageMeter()
        mask_probs = AverageMeter()
        lrs = AverageMeter()
        # un_labels = list(combined_list.keys())
        # class_counts = list(combined_list.values())
        # # sample_list = create_batches(predicted_list, batch_size=2)
        # unlabel_sampler = RFSSampler(un_labels, class_counts)
        # train_unlab_dataloader = DataLoader(train_unlab_dataset, batch_size=1 * 2, sampler=unlabel_sampler)
        print("Epoch:", epoch)
        unsup_iter = iter(train_unlab_dataloader_ori)  # unlab的第一个batch
        trainloader_iter = iter(train_lab_dataloader)
        # p_bar = tqdm(range(len(train_unlab_dataloader)))
        p_bar = tqdm(range(len(train_unlab_dataloader_ori)))

        i = 0

        for i in range(len(train_unlab_dataloader_ori)):
            # 取有监督训练集未增强的那一部分
            try:
                batch_label = next(trainloader_iter)
                batch_size = batch_label[0]['input_ids'].size(0)
                input_ids = batch_label[0]['input_ids'].to(device)
                bbox = batch_label[0]['bbox'].to(device)
                pixel_values = batch_label[0]['pixel_values'].to(device)
                attention_mask = batch_label[0]['attention_mask'].to(device)
                # token_type_ids = batch_label[0]['token_type_ids'].to(device)
                labels = batch_label[0]['labels'].to(device)
            except StopIteration:
                trainloader_iter = iter(train_lab_dataloader)
                batch_label = next(trainloader_iter)

                batch_size = batch_label[0]['input_ids'].size(0)
                input_ids = batch_label[0]['input_ids'].to(device)
                bbox = batch_label[0]['bbox'].to(device)
                pixel_values = batch_label[0]['pixel_values'].to(device)
                attention_mask = batch_label[0]['attention_mask'].to(device)
                # token_type_ids = batch_label['token_type_ids'].to(device)
                labels = batch_label[0]['labels'].to(device)
            # print("labels", labels) # tensor([[-100,    0, -100,  ..., -100, -100, -100],
            #         [-100,    0, -100,  ..., -100, -100, -100],
            #         [-100,    4,    4,  ..., -100, -100, -100],
            #         [-100,    0,    0,  ..., -100, -100, -100]], device='cuda:0')
            # exit()

            data_time.update(time.time() - end)
            '''有监督'''
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(input_ids=input_ids,
                            bbox=bbox,
                            pixel_values=pixel_values,
                            attention_mask=attention_mask,
                            labels=labels)
            # print("labels", labels.shape)  # [4, 512]
            # print("labels", labels.view(-1).shape)  # [2048]
            # exit()
            targets_np = labels.view(-1)  # .detach().cpu().numpy()  # [2048]

            # outputs = net(inputs)
            outputs_np = outputs.hidden_states.view(-1, 128)  # .detach().cpu().numpy()  # [2048,128]
            # 这边是新增的
            ###
            # label_logits = class_label(outputs.logits)
            label_logits = outputs.logits
            # print("label_logits", label_logits.shape)# [4, 512, 30])
            loss_sup = loss_fct(label_logits[:, :, :22].view(-1, 22), labels.view(-1))
            # loss_sup = outputs.loss
            # print("loss_sup", loss_sup)
            y_pred_sup = label_logits.detach()  # 强增强预测
            softmax_y_pred_sup = torch.softmax(y_pred_sup / T, dim=-1)
            if epoch == 40:
                with torch.no_grad():
                    try:
                        batch_unsup_ori = next(unsup_iter)
                    except StopIteration:  # 异常处理
                        unsup_iter_ori = iter(train_unlab_dataloader_ori)
                        batch_unsup_ori = next(unsup_iter_ori)
                    print('Testing on labelled examples in the training data...')
                    # print("length", len(train_unlab_dataloader_ori))
                    # for i in range(len(train_unlab_dataloader_ori)):

                    batch_unsup_w, batch_unsup_s = batch_unsup_ori[1], batch_unsup_ori[2]  # 取无监督训练集弱增强和强增强的那部分
                    input_ids_w = batch_unsup_w['input_ids'].to(device)
                    bbox_w = batch_unsup_w['bbox'].to(device)
                    pixel_values_w = batch_unsup_w['pixel_values'].to(device)
                    attention_mask_w = batch_unsup_w['attention_mask'].to(device)
                    # labels_w = batch_unsup_w['labels'].to(device)

                    # forward pass
                    outputs_w_no = model(input_ids=input_ids_w,
                                         bbox=bbox_w,
                                         pixel_values=pixel_values_w,
                                         attention_mask=attention_mask_w,
                                         labels=None)

                    # outputs = net(inputs)
                    # outputs_np = outputs_w.hidden_states.view(-1, 128).detach().cpu().numpy()  # [2048,128]
                    # 表征层
                    unoutputs_list.append(outputs_w_no.hidden_states.view(-1, 128))  # .detach().cpu().numpy())
                    predicted_list.append(outputs_w_no.logits.to(device))
                    targets_list.append(targets_np)
                    outputs_list.append(outputs_np)
                    loss = loss_sup
                    # print("unoutputs_list", unoutputs_list)
                    # This is used by training samples of labeled
            elif epoch >=41:
            # elif epoch > 10:
                # unlabeled samples的部分
                try:
                    batch_unsup = next(unsup_iter)
                except StopIteration:  # 异常处理
                    unsup_iter = iter(train_unlab_dataloader_ori)
                    batch_unsup = next(unsup_iter)

                batch_unsup_w, batch_unsup_s = batch_unsup[1], batch_unsup[2]  # 取无监督训练集弱增强和强增强的那部分

                '''无监督强增强'''

                # strong aug
                input_ids_s = batch_unsup_s['input_ids'].to(device)
                bbox_s = batch_unsup_s['bbox'].to(device)
                pixel_values_s = batch_unsup_s['pixel_values'].to(device)
                attention_mask_s = batch_unsup_s['attention_mask'].to(device)
                labels_s = batch_unsup_s['labels'].to(device)
                # optimizer.zero_grad()

                # forward + backward + optimize
                outputs_s = model(input_ids=input_ids_s,
                                  bbox=bbox_s,
                                  pixel_values=pixel_values_s,
                                  attention_mask=attention_mask_s,
                                  labels=labels_s)
                # targets_s = sinkhorn(outputs_s)

                # prediction for unlabelled data
                y_pred_unsup_s = outputs_s.logits.detach()  # 强增强预测
                outputs_s_hidden_embeding = outputs_s.hidden_states
                # y_pred_unsup_m = torch.mean(y_pred_unsup, 1)
                softmax_y_pred_unsup_s = torch.softmax(y_pred_unsup_s / T, dim=-1)

                '''无监督弱增强'''
                input_ids_w = batch_unsup_w['input_ids'].to(device)
                bbox_w = batch_unsup_w['bbox'].to(device)
                pixel_values_w = batch_unsup_w['pixel_values'].to(device)
                attention_mask_w = batch_unsup_w['attention_mask'].to(device)
                labels_w = batch_unsup_w['labels'].to(device)

                ## hidden 当做 projection header
                with torch.no_grad():
                    outputs_w = model(input_ids=input_ids_w,
                                      bbox=bbox_w,
                                      pixel_values=pixel_values_w,
                                      attention_mask=attention_mask_w,
                                      labels=labels_w)
                    # 用来鼓励学习扰动不变特征
                    outputs_w_o = outputs_w.logits
                    outputs_s_o = outputs_s.logits
                targets_s = torch.softmax(outputs_s_o[:, 0, :], dim=1)
                targets_w = torch.softmax(outputs_w_o[:, 0, :], dim=1)

                # prediction for unlabelled data
                y_pred_unsup_w = outputs_w.logits.detach()  # 弱增强预测
                outputs_w_hidden_embeding = outputs_w.hidden_states
                outputs_w_hidden = outputs_w_hidden_embeding.view(-1, 128)  # .detach().cpu().numpy()

                softmax_y_pred_unsup_w = torch.softmax(y_pred_unsup_w / T, dim=-1)
                # print(softmax_y_pred_unsup_w.shape)   #torch.Size([4, 512, 7])

                max_probs, target_u = torch.max(softmax_y_pred_unsup_w, dim=-1)  # values,indices
                pseudo = F.one_hot(target_u, num_classes=num_labels)

                # Fixmatch中的固定阈值

                # filter_pseudo_labels = torch.where(max_probs > threshold, target_u, -100)
                mask = max_probs.ge(threshold).float()

                targets = torch.cat(targets_list, dim=0)
                # print("targets", targets.shape)  # [389120]
                # outputs_l = np.concatenate(outputs_list, axis=0).astype(np.float64)
                outputs_l = torch.cat(outputs_list, dim=0)  # .to(dtype=torch.float64)
                # print("outputs_l", outputs_l)
                # un_output = np.concatenate(unoutputs_list, axis=0).astype(np.float64)
                un_output = torch.cat(unoutputs_list, dim=0)  # .to(dtype=torch.float64)
                # print("un_outputs_l_ddddddddddddd", un_output.shape)  #[389120, 128]

                cluster_centers_label = distribution(outputs_l, targets)  # [20,128]

                # 计算不同类别簇中心点之间的距离并保存成矩阵
                # un_outputs = torch.from_numpy(un_output).float().to(device)
                un_outputs = un_output.to(device)
                queue = FeatureQueue(classwise_max_size=None, temp_class=cluster_centers_label,
                                     initial_prototypes=cluster_centers_label, bal_queue=True)
                queue_ema = FeatureQueue_s(classwise_max_size=None, bal_queue=True, num_labels=num_labels)

                with torch.no_grad():  # memory里存的是ema的feat
                    # 已经使用了hidden_state
                    l_feats = ema_model.ema(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask,
                                            pixel_values=pixel_values, labels=labels).hidden_states
                    queue.enqueue(l_feats.view(-1, 128), labels.clone().detach().view(-1))
                    queue_ema.enqueue(l_feats.view(-1, 128), labels.clone().detach().view(-1))

                prototypes = queue.prototypes  # (K, D)
                prototypes_ema = queue_ema.prototypes
                # prototypes_w = queue_w.prototypes
                Dynamic_thres.update(probs_x_lb=softmax_y_pred_sup.view(-1, num_labels),
                                     probs_x_ulb=softmax_y_pred_unsup_w.view(-1,
                                                                             num_labels))  # 有标记样本上的分布，方便后面对无标记样本做出指导,无标记样本上的分布

                pseudo, mask, probs_x_ulb_reverse, probs_x_ulb_s_reverse, target_u, factor, distri_ = Dynamic_thres.masking(
                    probs_x_ulb=softmax_y_pred_unsup_w.view(-1, num_labels),
                    probs_x_lb=softmax_y_pred_sup.view(-1, num_labels),
                    probs_x_ulb_s=softmax_y_pred_unsup_s.view(-1, num_labels))
                class_prior = torch.linspace(1, 0, steps=30).to(device)
                class_prior = class_prior / class_prior.sum()
                epsilon = 1e-8
                class_prior += epsilon
                softmax_outputs = F.softmax(softmax_y_pred_unsup_s[:, 0, :], dim=1)
                log_softmax_outputs = torch.log(softmax_outputs + epsilon)
                prior_loss = F.kl_div(log_softmax_outputs, class_prior, reduction='batchmean')
                # if 10 < epoch < 20:
                #
                #     loss_unsup_no_weight = F.l1_loss(softmax_y_pred_unsup_s.view(-1, num_labels),
                #                                      pseudo.view(-1, num_labels),
                #                                      reduction='none')
                #     loss_unsup = loss_unsup_no_weight.sum(dim=-1).mean() + prior_loss.mean()
                # elif 20<= epoch < warmup_epochs:
                #     distances_list = pairwise_distances(un_outputs, prototypes)
                #     distances_list = distances_list.detach().cpu().numpy()
                #     gmm = GaussianMixture(n_components=30, max_iter=30, tol=1e-2, reg_covar=5e-4)
                #
                #     gmm.fit(distances_list)
                #     # 预测无标注数据的类别 格式是一个list
                #     predicted_labels = gmm.predict(distances_list)
                #     # 计算轮廓系数
                #     check_interval = 20
                #     if i % check_interval == 0:
                #         # 选择一个数据子集进行 Calinski-Harabasz 指数的计算
                #         subset_size = 1000  # 根据您的数据量调整这个数值
                #         indices = np.random.choice(len(distances_list), subset_size, replace=False)
                #         distances_subset = distances_list[indices]
                #         labels_subset = predicted_labels[indices]
                #
                #         # silhouette_avg = silhouette_score(distances_list, predicted_labels)
                #         calinski_avg = calinski_harabasz_score(distances_subset, labels_subset)
                #
                #         if calinski_avg > best_calinski_harabasz:
                #             best_calinski_harabasz = calinski_avg
                #             best_labels = predicted_labels
                #     # print("flattened_list_s", flattened_list_s)
                #     predicted_unlabel_dict = Counter(best_labels)
                #     predicted_unlabels_list = list(
                #         predicted_unlabel_dict.values())  # labels number of different classes
                #     print("predicted_unlabels_list", predicted_unlabels_list)
                #     if len(predicted_unlabels_list) < num_labels:
                #         # 如果列表长度小于目标长度，则添加1
                #         predicted_unlabels_list.extend([1] * (num_labels - len(predicted_unlabels_list)))
                #
                #     predicted_unlabels_list_a = np.array(
                #         [sum(predicted_unlabels_list) / x for x in predicted_unlabels_list])
                #     # proportions = [np.log((sum(my_list) / x + i) + 1e-6) for x in my_list]
                #     normalized_class_weight_s = np.log(predicted_unlabels_list_a)
                #
                #     current_iteration = (epoch - 19) * len(train_unlab_dataloader_ori) + i
                #     decay_factor = current_iteration / max_iteraion
                #     # 更新权重 - 这里使用线性衰减
                #     # 对于更复杂的策略，可以使用非线性函数
                #     # normalized_class_weight = (1 - decay_factor) * normalized_class_weight
                #     normalized_class_weight = (1 - decay_factor) * normalized_class_weight_s
                #     normalized_class_weight = torch.tensor(normalized_class_weight, dtype=torch.float).to(device)
                #
                #     loss_unsup_weight_mask = loss_func(softmax_y_pred_unsup_s.view(-1, num_labels), pseudo,
                #                                                 normalized_class_weight)
                #     loss_unsup = loss_unsup_weight_mask.mean() + prior_loss.mean()
                # else:
                #     distances_list = pairwise_distances(un_outputs, prototypes)
                #     distances_list = distances_list.detach().cpu().numpy()
                #     gmm = GaussianMixture(n_components=30, max_iter=30, tol=1e-2, reg_covar=5e-4)
                #     max_iteraion_s = (num_train_epochs - warmup_epochs + 1) * len(train_unlab_dataloader_ori)
                #
                #     gmm.fit(distances_list)
                #     # 预测无标注数据的类别 格式是一个list
                #     predicted_labels = gmm.predict(distances_list)
                #     # 计算轮廓系数
                #     check_interval = 20
                #     if i % check_interval == 0:
                #         # 选择一个数据子集进行 Calinski-Harabasz 指数的计算
                #         subset_size = 1000  # 根据您的数据量调整这个数值
                #         indices = np.random.choice(len(distances_list), subset_size, replace=False)
                #         distances_subset = distances_list[indices]
                #         labels_subset = predicted_labels[indices]
                #
                #         # silhouette_avg = silhouette_score(distances_list, predicted_labels)
                #         calinski_avg = calinski_harabasz_score(distances_subset, labels_subset)
                #
                #         if calinski_avg > best_calinski_harabasz:
                #             best_calinski_harabasz = calinski_avg
                #             best_labels = predicted_labels
                #     # print("flattened_list_s", flattened_list_s)
                #     predicted_unlabel_dict = Counter(best_labels)
                #
                #     predicted_unlabels_list = list(
                #         predicted_unlabel_dict.values())  # labels number of different classes
                #     print("predicted_unlabels_list", predicted_unlabels_list)
                #
                #     if len(predicted_unlabels_list) < num_labels:
                #         # 如果列表长度小于目标长度，则添加1
                #         predicted_unlabels_list.extend([1] * (num_labels - len(predicted_unlabels_list)))
                #
                #     predicted_unlabels_list_a = np.array(
                #         [sum(predicted_unlabels_list) / x for x in predicted_unlabels_list])
                #     # proportions = [np.log((sum(my_list) / x + i) + 1e-6) for x in my_list]
                #     normalized_class_weight_s = np.log(predicted_unlabels_list_a)
                #
                #     current_iteration = (epoch - warmup_epochs +1) * len(train_unlab_dataloader_ori) + i
                #     decay_factor = current_iteration / max_iteraion_s
                #     # 更新权重 - 这里使用线性衰减
                #     # 对于更复杂的策略，可以使用非线性函数
                #     # normalized_class_weight = (1 - decay_factor) * normalized_class_weight
                #     normalized_class_weight = (1 - decay_factor) * normalized_class_weight_s
                #     normalized_class_weight = torch.tensor(normalized_class_weight, dtype=torch.float).to(device)
                #     loss_unsup_weight_mask = loss_func_mask(softmax_y_pred_unsup_s.view(-1, num_labels), pseudo,
                #                                             normalized_class_weight, mask)
                #     loss_unsup = loss_unsup_weight_mask.mean() + prior_loss.mean()
                #     # loss_unsup_weight = loss_func(softmax_y_pred_unsup_s, pseudo, normalized_class_weight, num_labels)
                #     # loss_unsup = (loss_unsup_weight.sum(dim=-1) * mask.view(-1)).mean() + prior_loss.mean()
                # ###### new_bufen
                if epoch >= 41:
                    distances_list = pairwise_distances(un_outputs, prototypes)
                    distances_list = distances_list.detach().cpu().numpy()
                    gmm = GaussianMixture(n_components=30, max_iter=30, tol=1e-2, reg_covar=5e-4)
                    max_iteraion_s = (num_train_epochs - warmup_epochs + 1) * len(train_unlab_dataloader_ori)

                    gmm.fit(distances_list)
                    # 预测无标注数据的类别 格式是一个list
                    predicted_labels = gmm.predict(distances_list)
                    # 计算轮廓系数
                    check_interval = 20
                    if i % check_interval == 0:
                        # 选择一个数据子集进行 Calinski-Harabasz 指数的计算
                        subset_size = 1000  # 根据您的数据量调整这个数值
                        indices = np.random.choice(len(distances_list), subset_size, replace=False)
                        distances_subset = distances_list[indices]
                        labels_subset = predicted_labels[indices]

                        # silhouette_avg = silhouette_score(distances_list, predicted_labels)
                        calinski_avg = calinski_harabasz_score(distances_subset, labels_subset)

                        if calinski_avg > best_calinski_harabasz:
                            best_calinski_harabasz = calinski_avg
                            best_labels = predicted_labels
                    # print("flattened_list_s", flattened_list_s)
                    predicted_unlabel_dict = Counter(best_labels)

                    predicted_unlabels_list = list(
                        predicted_unlabel_dict.values())  # labels number of different classes
                    print("predicted_unlabels_list", predicted_unlabels_list)

                    if len(predicted_unlabels_list) < num_labels:
                        # 如果列表长度小于目标长度，则添加1
                        predicted_unlabels_list.extend([1] * (num_labels - len(predicted_unlabels_list)))

                    predicted_unlabels_list_a = np.array(
                        [sum(predicted_unlabels_list) / x for x in predicted_unlabels_list])
                    # proportions = [np.log((sum(my_list) / x + i) + 1e-6) for x in my_list]
                    normalized_class_weight_s = np.log(predicted_unlabels_list_a)

                    current_iteration = (epoch - warmup_epochs + 2) * len(train_unlab_dataloader_ori) + i
                    decay_factor = current_iteration / max_iteraion_s
                    # 更新权重 - 这里使用线性衰减
                    # 对于更复杂的策略，可以使用非线性函数
                    # normalized_class_weight = (1 - decay_factor) * normalized_class_weight
                    normalized_class_weight = (1 - decay_factor) * normalized_class_weight_s
                    normalized_class_weight = torch.tensor(normalized_class_weight, dtype=torch.float).to(device)
                    loss_unsup_weight_mask = loss_func_mask(softmax_y_pred_unsup_s.view(-1, num_labels), pseudo,
                                                            normalized_class_weight, mask)
                    loss_unsup = loss_unsup_weight_mask.mean() + prior_loss.mean()
                    # 转换为 one-hot 编码
                    num_classes = prototypes_ema.size(0)  # 假设原型的数量即类别数
                    similarity_matrix_w = F.cosine_similarity(outputs_w_hidden_embeding[:, 0, :].unsqueeze(1), prototypes_ema.unsqueeze(0),
                                                            dim=2)
                    similarity_matrix_w = torch.argmax(similarity_matrix_w, dim=1)


                    cosine_similarity_matrix_w = F.one_hot(similarity_matrix_w, num_classes=num_classes)

                    # scores_w = torch.mm(outputs_w_hidden_embeding[:, 0, :],
                    #                     prototypes_ema.t())  # [4,128]*[128,30]_>[4,30]
                    similarity_matrix_s = F.cosine_similarity(outputs_s_hidden_embeding[:, 0, :].unsqueeze(1), prototypes_ema.unsqueeze(0),
                                                            dim=2)
                    similarity_matrix_s = torch.argmax(similarity_matrix_s, dim=1)

                    # 转换为 one-hot 编码
                    cosine_similarity_matrix_s = F.one_hot(similarity_matrix_s, num_classes=num_classes)

                    # scores_s = torch.mm(outputs_s_hidden_embeding[:, 0, :],
                    #                     prototypes_ema.t())  # [4,128]*[128,30]_>[4,30]

                    alpha = 1.0
                    l = np.random.beta(alpha, alpha)
                    # print("targets_s", targets_s.shape) # [4, 30]
                    # targets_w 是softmax
                    # mixed_target_s = l * targets_w + (1 - l) * scores_s  # +
                    mixed_target_w = F.normalize(targets_w * cosine_similarity_matrix_w, p=1)  # +
                    # mixed_target_w = l * targets_s + (1 - l) * scores_w  # +
                    mixed_target_s = F.normalize(targets_s * cosine_similarity_matrix_s, p=1)

                    diffusion_params = {
                        'diffusion_steps': 1,
                        'q': 0.8
                    }
                    k = 5
                    # z_features_s = outputs_s_hidden_embeding[:, 0, :]
                    # z_features_s = F.normalize(z_features_s, dim=-1)
                    #
                    # z_features_w = outputs_w_hidden_embeding[:, 0, :]
                    # z_features_w = F.normalize(z_features_w, dim=-1)

                    # 这个对应的就是   没有head  强
                    prompt_features_s = outputs_s_hidden_embeding[:, :5, :][:, 1:2 + 1, :].mean(dim=1)

                    # 这个对应的就是有没有head   弱 no_grad
                    prompt_features_w = outputs_w_hidden_embeding[:, :5, :][:, 1:2 + 1, :].mean(dim=1)

                    # 这个对应的就是   有head    强
                    pp_features_s = F.normalize(prompt_features_s, dim=-1)
                    pp_features_w = F.normalize(prompt_features_w, dim=-1)

                    # z_features_ss = l * z_features_w + (1 - l) * z_features_s
                    # z_features_w_me = l * z_features_s + (1 - l) * z_features_w

                    z_pp_features_s = l * pp_features_w + (1 - l) * pp_features_s
                    # 这个对应的就是没有head 弱 no_grad
                    z_pp_prompt_w_me = l * pp_features_s + (1 - l) * pp_features_w
                    # 这个对应的就是有head 弱 no_grad

                    simhash_pp = SimHashTorch(num_bits=128)
                    simhash_cls = SimHashTorch(num_bits=30)
                    '''
                    [z_features_s, z_features_ss, pp_features_s, z_pp_features_s]
                    [z_features_w, z_features_w_me, pp_features_w, z_pp_prompt_w_me]
                    '''
                    # '''
                    ## pKNN from ppfeatures
                    pknn_pp_features_me = torch.cat([f for f in z_pp_prompt_w_me.chunk(2)][::-1], dim=0)
                    # membank_unsup_z.add(v=z_features_w_me, y=None)
                    # pp_features_mb, _ = membank_unsup_dpr_z.query()
                    pp_features_mb = z_pp_features_s.to(device)  # 我们这里就不用memory，而是交叉混合

                    q_pp_features = pp_features_s
                    k_pp_features_mb = pp_features_mb
                    k_pp_features_me = pknn_pp_features_me

                    pp_similarity, pp_knn_affinity, pp_transition_prob = compute_pseudo_knn(pknn_pp_features_me,
                                                                                            pp_features_mb,
                                                                                            diffusion_params=diffusion_params,
                                                                                            k=k,
                                                                                            )
                    # print("pp_knn_affinity", pp_knn_affinity.shape) # [4, 4]
                    pp_neg_affinity = negative_sampling_from_membank(pp_knn_affinity, similarity=pp_similarity,
                                                                     neg_samples=1024)

                    # 创建 SimHash 实例
                    # 更新 KNN 亲和力
                    pp_knn_affinity, pp_neg_affinity = update_knn_affinity_with_simhash_torch(pp_similarity,
                                                                                              pp_knn_affinity,
                                                                                              pp_neg_affinity,
                                                                                              pknn_pp_features_me,
                                                                                              simhash_pp)

                    ### pKNN loss on pp
                    pp_knn_contrastive_loss = compute_knn_loss(pp_knn_affinity, pp_neg_affinity, q_pp_features,
                                                               k_pp_features_mb, k_pp_features_me,
                                                               temperature=0.07, k=6,
                                                               epoch=epoch,
                                                               )
                    print("pp_knn_contrastive_loss", pp_knn_contrastive_loss)

                    ### PCL from cls
                    ### PCL from cls
                    pknn_cc_features_me = torch.cat([f for f in mixed_target_w.chunk(2)][::-1], dim=0)
                    # targets_w
                    features_cls_mb = mixed_target_s.to(device)
                    # q_features = z_features  # B,D
                    k_features_cls_mb = features_cls_mb
                    kk_features_me = mixed_target_w
                    similarity, knn_affinity, transition_prob = compute_pseudo_knn(
                        pknn_cc_features_me,
                        features_cls_mb,
                        diffusion_params=diffusion_params,
                        k=k)

                    neg_affinity = negative_sampling_from_membank(knn_affinity, similarity=similarity, neg_samples=128)
                    # 跟新的knn_affinity
                    knn_affinity, neg_affinity = update_knn_affinity_with_simhash_torch(similarity,
                                                                                        knn_affinity,
                                                                                        neg_affinity,
                                                                                        pknn_cc_features_me,
                                                                                        simhash_cls)
                    # calculate compute_knn_loss for cls
                    # knn_contrastive_loss = 0.00 第四个位置参数mixed_target_s与compute_pseudo_knn的参数一致 mocco
                    # 第三个位置的参数要与是 第四个位置的强增广， 第五个位置的mixed_target_w与pknn_pp_features_me的cat里面的一致
                    knn_contrastive_loss = compute_knn_loss(knn_affinity, neg_affinity, targets_s, k_features_cls_mb,
                                                            kk_features_me,
                                                            temperature=0.07, k=5, epoch=epoch,
                                                            )
                    print("knn_contrastive_loss", knn_contrastive_loss)
                    # '''

                else:
                    pp_knn_contrastive_loss = torch.tensor(0.0, device=device)
                    knn_contrastive_loss = torch.tensor(0.0, device=device)

                '''
                # generate hard pseudo-labels for confident novel class samples
                # targets_u_novel = targets_u[:, args.no_seen:]
                targets_u_novel = mixed_target_w[:, 22:]
                max_pred_novel, _ = torch.max(targets_u_novel, dim=-1)
                hard_novel_idx1 = torch.where(max_pred_novel >= 0.95)[0]

                # targets_u2_novel = targets_u2[:, args.no_seen:]
                targets_s_novel = mixed_target_s[:, 22:]
                max_preds_novel, _ = torch.max(targets_s_novel, dim=-1)
                hard_novel_idx2 = torch.where(max_preds_novel >= 0.95)[0]

                targets_w[hard_novel_idx1] = targets_w[hard_novel_idx1].ge(0.95).float()
                targets_s[hard_novel_idx2] = targets_s[hard_novel_idx2].ge(0.95).float()

                # mixup
                all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
                all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
                all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
                all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
                all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)

                all_targets = torch.cat([targets_x, targets_w, targets_s], dim=0)
                # all_temp = torch.cat([temp_x, temp_u, temp_u], dim=0)
                # alpha = 5
                # l = np.random.beta(alpha, alpha)
                #
                # idx = torch.randperm(all_inputs.size(0))
                #
                # input_a, input_b = all_inputs, all_inputs[idx]
                # target_a, target_b = all_targets, all_targets[idx]
                #
                # mixed_input = l * input_a + (1 - l) * input_b
                # mixed_target = l * target_a + (1 - l) * target_b

                # interleave labeled and unlabled samples between batches to get correct batchnorm calculation
                mixed_input = list(torch.split(mixed_input, batch_size))
                mixed_input = interleave(mixed_input, batch_size)

                logits = [model(mixed_input[0])]
                for input in mixed_input[1:]:
                    logits.append(model(input))

                # put interleaved samples back
                logits = interleave(logits, batch_size)
                logits_x = logits[0]
                logits_u = torch.cat(logits[1:], dim=0)
                logits = torch.cat((logits_x, logits_u), 0)

                # cross_entropy loss
                # preds = F.log_softmax(logits / mixed_temp.unsqueeze(1), dim=1)
                preds = F.log_softmax(logits, dim=1)
                loss_d2d = -torch.mean(torch.sum(mixed_target * preds, dim=1))
                '''
                print("loss_unsup", loss_unsup)
                ### compute loss weight for pknn
                w_knn_loss = annealing_linear_ramup(0, 0.5, i,
                                                    2 * len(train_unlab_dataloader_ori))
                # loss = loss_sup + 0.1 * loss_unsup + 0.05 * loss_d2d
                loss = loss_sup + 0.15 * loss_unsup + 0.15 * ((1 - w_knn_loss) * pp_knn_contrastive_loss + w_knn_loss * knn_contrastive_loss)  # + 0.2 * loss_imbalance # + 0.12 * knn_contrastive_loss# + 0.05 * loss_d2d
                # loss = loss_sup + alpha * loss_d2d
                # print("loss", loss)
                losses.update(loss.item())  # textboardx loss可视化
                losses_x.update(loss_sup.item())
                losses_u.update(loss_unsup.item())
                losses_imbalance.update(pp_knn_contrastive_loss.item())
                losses_cls.update(knn_contrastive_loss.item())
                # losses_d2d.update(loss_d2d.item())
                batch_time.update(time.time() - end)
                end = time.time()
                # mask_probs.update(mask.mean().item())
                p_bar.set_description(
                    # "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Loss_d2d: {loss_d2d:.4f}. Mask: {mask:.2f}. ".format(
                    # "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. loss_imbalance: {loss_imbalance:.4f}. ".format(
                    "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. loss_imbalance: {loss_imbalance:.4f}. loss_cls: {loss_cls:.4f}".format(
                        epoch=epoch + 1,
                        epochs=num_train_epochs,
                        batch=i + 1,
                        iter=len(train_unlab_dataloader_ori),
                        # lr=scheduler.get_last_lr()[0],
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        loss_x=losses_x.avg,
                        loss_u=losses_u.avg,
                        loss_imbalance=losses_imbalance.avg,
                        loss_cls=losses_cls.avg
                    ))
                # mask=mask_probs.avg))
            else:
                loss = loss_sup
            # with open('unoutputs_list.txt', 'w') as file:
            #         for sublist in unoutputs_list:
            #             line = ' '.join(map(str, sublist))  # 将每个子列表的元素转换为字符串并用空格连接
            #             file.write(f"{line}\n")
            # with open('predicted_list.txt', 'w') as file:
            #         for sublist in predicted_list:
            #             line = ' '.join(map(str, sublist))  # 将每个子列表的元素转换为字符串并用空格连接
            #             file.write(f"{line}\n")
            loss.backward()
            losses.update(loss.item())
            # lrs.update(scheduler.get_last_lr()[0])
            optimizer.step()
            # model.ema_update()  # 更新教师模型
            if epoch > 40:
                predicted_list_temp = PredictionsQueue(max_length=(len(unoutputs_list)),
                                                       queue=unoutputs_list)
                predicted_list_temp.update(outputs_w_hidden)
                unoutputs_list_s = [item for item in predicted_list_temp.queue]
                unoutputs_list = unoutputs_list_s
                # membank_lmask.add(v=mask_lab.repeat(2).view(-1, 1).float(), y=None)
            if epoch > 40 and len(membank_label) + 1:
                membank_label.add(v=labels[:, 0].view(-1).repeat(2).view(-1, 1).float(), y=None)
                # membank_unsup_z.add(v=z_features_w_me, y=None)
            with torch.no_grad():
                m = 1 - (1 - 0.995) * (math.cos(math.pi * epoch / num_train_epochs) + 1) / 2
                # ema_model = model.ema_update()
                # ema_model=model.teacher, model=model.student, alpha_teacher=m, iteration=epoch)
                ema_model.update(model, m)
            # mask_probs.update(mask.mean().item())
            '''打印progress'''
            if epoch <= 40:
                p_bar.set_description(
                    "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. ".format(
                        epoch=epoch + 1,
                        epochs=num_train_epochs,
                        batch=i + 1,
                        iter=len(train_unlab_dataloader_ori),
                        # lr=scheduler.get_last_lr()[0],
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg))
            p_bar.update()
        p_bar.close()
        writer = SummaryWriter(out)
        writer.add_scalar('train/1.train_loss', losses.avg, epoch)
        writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
        writer.add_scalar('train/3.train_loss_u', losses_u.avg, epoch)
        writer.add_scalar('train/4.train_losses_imbalance', losses_imbalance.avg, epoch)
        writer.add_scalar('train/5.train_loss_cls', losses_cls.avg, epoch)

        test_model = ema_model.ema
        if epoch > 0 and epoch % 5 == 0:
            preds_val, out_label_ids, test_loss = test(test_dataloader, test_model)
            test_result, test_class_report = results_test(preds_val, out_label_ids)
            precision = test_result['precision']
            recall = test_result['recall']
            f1 = test_result['f1']
            accuracy = test_result['accuracy']
            test_losses.update(test_loss)

            # writer.add_scalar('train/6.mask', mask_probs.avg, epoch)
            # writer.add_scalar('train/5.lrs', scheduler.get_last_lr()[0], epoch)
            writer.add_scalar('test/1.test_loss', test_losses.avg, epoch)
            writer.add_scalar('test/2.f1', f1, epoch)
            writer.add_scalar('test/3.precision', precision, epoch)
            writer.add_scalar('test/4.recall', recall, epoch)
            writer.add_scalar('test/5.accuracy', accuracy, epoch)

            is_best = f1 > best_f1
            # if is_best and epoch > 15:
            # if epoch > 10 and epoch % 5 == 0:
            #     rel_precision = precision
            #     rel_recall = recall
            #     rel_accuracy = accuracy
            #     embeddings = outputs_w_hidden_embeding.view(-1, 128).detach().cpu().numpy()
            #     _, y = torch.max(softmax_y_pred_unsup_w.view(-1, num_labels), dim=-1)  # values,indices
            #     y = y.cpu().numpy()
            #     #########添加了一个绘制u_w的t-NSE的图
            #     plot_TSNE(embeddings, y, out + "/TSNE_embedding_after.png")

            best_f1 = max(f1, best_f1)

            model_to_save = model.module if hasattr(model, "module") else model

            ema_to_save = ema_model.ema.module if hasattr(
                ema_model.ema, "module") else ema_model.ema

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict(),
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'best_f1': best_f1,
                'rel_precision': rel_precision,
                'rel_recall': rel_recall,
                'rel_accuracy': rel_accuracy,
                # 'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                # 'scheduler': scheduler.state_dict(),
            }, is_best, out)

            test_f1s.append(f1)
            print(test_f1s)
            logger.info('################################################################')
            logger.info('Best top-1 f1: {:.6f}'.format(best_f1))
            logger.info('Mean top-1 f1: {:.6f}\n'.format(
                np.mean(test_f1s)))  #########不能直接输出列表
            ### 上述已经算过平均值
            logger.info('Best top-1 f1 relative precision: {:.6f}'.format(rel_precision))
            logger.info('Best top-1 f1 relative recall: {:.6f}'.format(rel_recall))
            logger.info('Best top-1 f1 relative accuracy: {:.6f}'.format(rel_accuracy))
            logger.info(f'test_class_report:{test_class_report}')
        writer.close()


def test(test_dataloader, model):  ########这里删掉epoch
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # losses = AverageMeter()
    end = time.time()
    preds_val = None
    out_label_ids = None

    loop = tqdm(test_dataloader, total=len(test_dataloader), leave=True)
    eval_losses = 0.0
    nb_eval_steps = 0
    prediction_list = []
    labels_list = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loop):
            data_time.update(time.time() - end)
            model.eval()
            input_ids = batch['input_ids'].to(device)
            bbox = batch['bbox'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            # labels_list.extend(labels)

            # forward pass
            outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask,
                            pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            ##########################下面是我改的
            # losses.update(loss.item(), batch['input_ids'].shape[0])
            eval_losses += loss.item()
            nb_eval_steps += 1
            batch_time.update(time.time() - end)
            # print("predictions", predictions)
            if preds_val is None:
                preds_val = outputs.logits.detach().cpu().numpy()
                out_label_ids = batch["labels"].detach().cpu().numpy()
            else:
                preds_val = np.append(preds_val, outputs.logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, batch["labels"].detach().cpu().numpy(), axis=0
                )
                # 画出协方差图

    return preds_val, out_label_ids, eval_losses / nb_eval_steps


if __name__ == '__main__':
    train(train_lab_dataloader, train_unlab_dataloader_ori, train_unlab_dataset, test_dataloader,
          model, optimizer, ema_model)  # 在这里把scheduler去掉了