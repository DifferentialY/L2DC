import argparse
import time
import torch
import torch.nn.functional as F
import logging
import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import random
import utils
import scipy.sparse as sparse
from models import *
from l2dc_inference import both_infer
from data_loader import loader
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score
from accuracy import clustering_accuracy

parser = argparse.ArgumentParser(description='SURE in PyTorch')
parser.add_argument('--data', default='0', type=int,
                    help='choice of dataset, 0-Scene15, 1-Caltech101, 2-Reuters10, 3-NoisyMNIST,'
                         '4-DeepCaltech, 5-DeepAnimal, 6-MNISTUSPS')
parser.add_argument('-li', '--log-interval', default='1', type=int, help='interval for logging info')
parser.add_argument('-bs', '--batch-size', default='1024', type=int, help='number of batch size')
parser.add_argument('-e', '--epochs', default='80', type=int, help='number of epochs to run')
parser.add_argument('-me', '--mse_epochs', default='0', type=int, help='the epoch which AE begins')
parser.add_argument('-ce', '--con_epochs', default='80', type=int, help='the epoch which contrastive begins')
parser.add_argument('-te', '--tune_epochs', default='0', type=int, help='the epoch which fine_tuning begins')
parser.add_argument('-lr', '--learn-rate', default='1e-3', type=float, help='learning rate of adam')
parser.add_argument('--lam1', default='1', type=float, help='hyper-parameter of noise contrastive loss')
parser.add_argument('--lam2', default='1', type=float, help='hyper-parameter of noise contrastive estimation loss')
parser.add_argument('-noise', '--noisy-training', type=bool, default=True,
                    help='training with noisy negatives')
parser.add_argument('-np', '--neg-prop', default='30', type=int, help='the ratio of negative to positive pairs')
parser.add_argument('--edge', default='1.0', type=float, help='the edging')
parser.add_argument('-m', '--margin', default='5', type=int, help='initial margin')
parser.add_argument('-sm', '--simi_margin', default='0.2', type=float, help='xx')
parser.add_argument('--temperature', default='0.5', type=float, help='the temperature parameter')
parser.add_argument('--hard', default=True, type=bool, help='use hard contrastive or not')
parser.add_argument('--gpu', default='0', type=str, help='GPU device idx to use.')
parser.add_argument('-r', '--robust', default=True, type=bool, help='use our robust loss or not')
parser.add_argument('-sc', '--semantic_con', default=True, type=bool, help='semantic contrastive or latent feature contrastive')
parser.add_argument('-er', '--entropy_regular', default=True, type=bool, help='Entropy regularization or not')
parser.add_argument('-sn', '--similarity_normalized', default=True, type=bool, help='Normalize the similarity distance or not')
parser.add_argument('-et', '--Eswitching-time', default=1.0, type=float, help='start fine when neg_dist>=et*margin')
parser.add_argument('-ct', '--Cswitching-time', default=1.0, type=float, help='start fine when neg_cosine<=ct*simi_margin')
parser.add_argument('-s', '--start-fine', default=False, type=bool, help='flag to start use robust loss or not')
parser.add_argument('--settings', default=2, type=int, help='0-PVP, 1-PSP, 2-Both')
parser.add_argument('-ap', '--aligned-prop', default='1.0', type=float,
                    help='originally aligned proportions in the partially view-unaligned data')
parser.add_argument('-cp', '--complete-prop', default='1.0', type=float,
                    help='originally complete proportions in the partially sample-missing data')
parser.add_argument('-mr', '--measure_realign', default='0', type=int,
                    help='choice of mesurement of realignmen'
                         't, 0-Euclidean, 1-Cosine')

args = parser.parse_args()
# print("==========\nArgs:{}\n==========".format(args))
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# mean distance of four kinds of pairs, namely, pos., neg., true neg., and false neg. (noisy labels)
pos_dist_mean_list, neg_dist_mean_list, true_neg_dist_mean_list, false_neg_dist_mean_list = [], [], [], []
pos_cosine_mean_list, neg_cosine_mean_list, true_neg_cosine_mean_list, false_neg_cosine_mean_list = [], [], [], []


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    u = linear_sum_assignment(w.max() - w)
    ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels) + 1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)


class NoiseRobustLoss(nn.Module):
    def __init__(self):
        super(NoiseRobustLoss, self).__init__()

        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N // 2):
            mask[i, N // 2 + i] = 0
            mask[N // 2 + i, i] = 0
        return mask

    def forward_feature(self, pair_dist, P, margin, use_robust_loss, args):
        dist_sq = pair_dist * pair_dist
        P = P.to(torch.float32)
        N = len(P)
        if use_robust_loss == 1:
            if args.start_fine:
                loss = P * dist_sq + (1 - P) * (1 / margin) * torch.pow(
                    torch.clamp(torch.pow(pair_dist, args.edge) * (margin - pair_dist), min=0.0), 2)
            else:
                loss = P * dist_sq + (1 - P) * torch.pow(
                    torch.clamp(margin - pair_dist, min=0.0), 2)
        else:
            loss = P * dist_sq + (1 - P) * torch.pow(
                torch.clamp(margin - pair_dist, min=0.0), 2)
        loss = torch.sum(loss) / (2.0 * N)
        return loss

    def forward_label(self, q_i, q_j, P, args):
        pos_index = (P == 1).nonzero()
        pos_index = pos_index.reshape(len(pos_index))
        q_i = q_i[pos_index]
        q_j = q_j[pos_index]
        n, class_num = q_i.shape

        if args.semantic_con == 0:
            N = 2 * n
            h = torch.cat((q_i, q_j), dim=0)

            sim = self.similarity(h.unsqueeze(1), h.unsqueeze(0)) / args.temperature
            sim_i_j = torch.diag(sim, n)
            sim_j_i = torch.diag(sim, -n)

            positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
            mask = self.mask_correlated_samples(N)
            labels = torch.zeros(N).to(positive_samples.device).long()

            if args.hard:
                loss = []
                hard_mask = torch.zeros(N, N)
                hard_mask = torch.where(sim.cpu() < args.simi_margin, hard_mask, mask)
                hard_mask = hard_mask.bool()
                for i in range(N):
                    negative_samples = sim[i, :][hard_mask[i, :]]
                    logits = torch.cat((positive_samples[i], negative_samples))
                    loss.append(self.criterion(logits, labels[i]))
                loss = sum(loss)
            else:
                mask = mask.bool()
                negative_clusters = sim[mask].reshape(N, -1)
                logits = torch.cat((positive_samples, negative_clusters), dim=1)
                loss = self.criterion(logits, labels)
            loss /= N
        else:
            q_i = q_i.t()
            q_j = q_j.t()
            N = 2 * class_num
            q = torch.cat((q_i, q_j), dim=0)

            if args.similarity_normalized:
                sim = self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / args.temperature
            else:
                sim = (torch.matmul(q, q.T) / args.temperature).to(q.device)
            sim_i_j = torch.diag(sim, class_num)
            sim_j_i = torch.diag(sim, -class_num)

            positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
            mask = self.mask_correlated_samples(N)
            mask = mask.bool()
            negative_clusters = sim[mask].reshape(N, -1)

            labels = torch.zeros(N).to(positive_clusters.device).long()
            logits = torch.cat((positive_clusters, negative_clusters), dim=1)
            loss = self.criterion(logits, labels)
            loss /= N
            if args.entropy_regular:
                p_i = q_i.t().sum(0).view(-1)
                p_i /= p_i.sum()
                p_i = p_i.masked_fill(p_i == 0, 1e-33)
                ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
                p_j = q_j.t().sum(0).view(-1)
                p_j /= p_j.sum()
                p_j = p_j.masked_fill(p_j == 0, 1e-33)
                ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
                entropy = ne_i + ne_j
                loss += entropy
        return loss


def train(train_loader, model, criterion, optimizer, epoch, args):
    pos_dist = 0  # mean distance of pos. pairs
    pos_cosine = 0

    neg_dist = 0
    neg_cosine = 0

    false_neg_dist = 0  # mean distance of false neg. pairs (pairs in noisy labels)
    false_neg_cosine = 0

    true_neg_dist = 0
    true_neg_cosine = 0

    pos_count = 0  # count of pos. pairs
    neg_count = 0
    false_neg_count = 0  # count of neg. pairs (pairs in noisy labels)
    true_neg_count = 0

    if epoch % args.log_interval == 0:
        logging.info("=======> Train epoch: {}/{}/{}".format(epoch, args.con_epochs - 1, args.mse_epochs + args.con_epochs + args.tune_epochs - 1))
    model.train()
    time0 = time.time()
    ncl_loss_value = 0
    ver_loss_value = 0
    nce_loss_value = 0
    loss_value = 0
    for batch_idx, (x0, x1, labels, real_labels, sample_labels, _, _) in enumerate(train_loader):
        # labels refer to noisy labels for the constructed pairs, while real_labels are the clean labels for these pairs
        x0, x1, labels, real_labels, sample_labels = x0.to(device), x1.to(device), labels.to(device), real_labels.to(
            device), sample_labels.to(device)
        x0 = x0.view(x0.size()[0], -1)
        x1 = x1.view(x1.size()[0], -1)
        try:
            h0, h1, pseudo0, pseudo1, z0, z1 = model(x0, x1)
        except:
            print("error raise in batch", batch_idx)

        pair_dist = F.pairwise_distance(h0, h1)
        pair_cosine = F.cosine_similarity(pseudo0, pseudo1, dim=1)

        pos_dist += torch.sum(pair_dist[labels == 1])
        pos_cosine += torch.sum(pair_cosine[labels == 1])

        neg_dist += torch.sum(pair_dist[labels == 0])
        neg_cosine += torch.sum(pair_cosine[labels == 0])

        true_neg_dist += torch.sum(pair_dist[torch.logical_and(labels == 0, real_labels == 0)])
        true_neg_cosine += torch.sum(pair_cosine[torch.logical_and(labels == 0, real_labels == 0)])

        false_neg_dist += torch.sum(pair_dist[torch.logical_and(labels == 0, real_labels == 1)])
        false_neg_cosine += torch.sum(pair_cosine[torch.logical_and(labels == 0, real_labels == 1)])

        pos_count += len(pair_dist[labels == 1])
        neg_count += len(pair_dist[labels == 0])
        true_neg_count += len(pair_dist[torch.logical_and(labels == 0, real_labels == 0)])
        false_neg_count += len(pair_dist[torch.logical_and(labels == 0, real_labels == 1)])

        ncl_loss = criterion[0].forward_feature(pair_dist, labels, args.margin, args.robust, args)
        nce_loss = criterion[0].forward_label(pseudo0, pseudo1, labels, args)
        ver_loss = criterion[1](x0, z0) + criterion[1](x1, z1)

        loss = args.lam1 * ncl_loss + args.lam2 * nce_loss + ver_loss

        ncl_loss_value += ncl_loss.item()
        nce_loss_value += nce_loss.item()
        ver_loss_value += ver_loss.item()
        loss_value += loss.item()

        if epoch != args.mse_epochs:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    epoch_time = time.time() - time0

    pos_dist /= pos_count
    pos_cosine /= pos_count

    neg_dist /= neg_count
    neg_cosine /= neg_count

    true_neg_dist /= true_neg_count
    true_neg_cosine /= true_neg_count

    false_neg_dist /= false_neg_count
    false_neg_cosine /= false_neg_count

    if epoch != args.mse_epochs and args.robust and neg_dist >= args.Eswitching_time * args.margin and not args.start_fine:
        # start fine when the mean distance of neg. pairs is greater than Eswitching_time * margin
        args.start_fine = True
        logging.info("******* neg_dist_mean >= {} * margin, start using fine loss at epoch: {} *******"
                     .format(args.Eswitching_time, epoch + 1))

    # margin = the pos. distance + neg. distance before training
    if args.data == 0:
        if epoch == args.mse_epochs + 1 and args.margin != 1.0:
            args.margin = max(1, round((pos_dist + neg_dist).item()))
            # args.margin = 5
            logging.info("margin = {}".format(args.margin))
    else:
        if epoch == args.mse_epochs and args.margin != 1.0:
            args.margin = max(1, round((pos_dist + neg_dist).item()))
            # args.margin = 5
            logging.info("margin = {}".format(args.margin))

    if epoch == args.mse_epochs and args.simi_margin != 0.1:
        args.simi_margin = min(0.1, neg_cosine.item())
        logging.info("simi_margin = {}".format(args.simi_margin))

    if epoch % args.log_interval == 0:
        logging.info(
            "dist: P = {}, N = {}, TN = {}, FN = {}; cosine: P = {}, N = {}, TN = {}, FN = {}; ncl_loss: {}, ver_loss:{}, nce_loss: {}, time = {} s"
            .format(round(pos_dist.item(), 2), round(neg_dist.item(), 2),
                    round(true_neg_dist.item(), 2), round(false_neg_dist.item(), 2),
                    round(pos_cosine.item(), 2), round(neg_cosine.item(), 2),
                    round(true_neg_cosine.item(), 2), round(false_neg_cosine.item(), 2),
                    round(ncl_loss_value / len(train_loader), 2),
                    round(ver_loss_value / len(train_loader), 2),
                    round(nce_loss_value / len(train_loader), 2),
                    round(epoch_time, 2)))
    LOSS = loss_value / len(train_loader)
    LOSS_ncl = ncl_loss_value / len(train_loader)
    LOSS_nce = nce_loss_value / len(train_loader)
    LOSS_ver = ver_loss_value / len(train_loader)

    return pos_dist, neg_dist, false_neg_dist, true_neg_dist, pos_cosine, neg_cosine, false_neg_cosine, true_neg_cosine, epoch_time, LOSS, LOSS_ncl, LOSS_nce, LOSS_ver


def make_pseudo_label(data, model, class_num, data_size, device):
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=data_size,
        shuffle=True
    )
    model.eval()
    scaler = MinMaxScaler()
    for batch_idx, (x0, x1, labels) in enumerate(loader):
        x0 = x0.to(device)
        x1 = x1.to(device)
        x0 = x0.view(x0.size()[0], -1)
        x1 = x1.view(x1.size()[0], -1)
        with torch.no_grad():
            # _, _, latent0, latent1, _, _, _, _ = model.forward(x0, x1)
            latent0, latent1, _, _, _, _ = model.forward(x0, x1)
        latent0 = latent0.cpu().detach().numpy()
        latent0 = scaler.fit_transform(latent0)
        latent1 = latent1.cpu().detach().numpy()
        latent1 = scaler.fit_transform(latent1)
        latent = np.concatenate((latent0, latent1), axis=1)

    kmeans = KMeans(n_clusters=class_num, n_init=100)
    new_pseudo_label = []
    Pseudo_label0 = kmeans.fit_predict(latent)
    Pseudo_label0 = Pseudo_label0.reshape(data_size, 1)
    Pseudo_label0 = torch.from_numpy(Pseudo_label0)
    new_pseudo_label.append(Pseudo_label0)
    Pseudo_label1 = Pseudo_label0
    new_pseudo_label.append(Pseudo_label1)

    return new_pseudo_label


class gettestdata(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        fea0, fea1 = (torch.from_numpy(self.data[0][:, index])).type(torch.FloatTensor), (
            torch.from_numpy(self.data[1][:, index])).type(torch.FloatTensor)
        fea0, fea1 = fea0.unsqueeze(0), fea1.unsqueeze(0)
        label = np.int64(self.labels[index])
        return fea0, fea1, label

    def __len__(self):
        return len(self.labels)


def match(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    new_y = np.zeros(y_true.shape[0])
    for i in range(y_pred.size):
        for j in row_ind:
            if y_true[i] == col_ind[j]:
                new_y[i] = row_ind[j]
    new_y = torch.from_numpy(new_y).long().to(device)
    new_y = new_y.view(new_y.size()[0])
    return new_y


def eva(label, pred):
    nmi = normalized_mutual_info_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    acc = cluster_acc(label, pred)
    pur = purity(label, pred)
    return nmi, ari, acc, pur


# def get_sparse_rep(senet, data, batch_size=10, chunk_size=100, non_zeros=1000):
def get_sparse_rep(C, data, non_zeros=1000):
    N, D = data.shape
    non_zeros = min(N, non_zeros)
    _, index = torch.topk(torch.abs(C), dim=1, k=non_zeros)
    val = C.gather(1, index).reshape([-1]).cpu().data.numpy()
    indicies = index.reshape([-1]).cpu().data.numpy()
    indptr = [non_zeros * i for i in range(N + 1)]
    C_sparse = sparse.csr_matrix((val, indicies, indptr), shape=[N, N])
    return C_sparse


def get_knn_Aff(C_sparse_normalized, k=3, mode='symmetric'):
    C_knn = kneighbors_graph(C_sparse_normalized, k, mode='connectivity', include_self=False, n_jobs=10)
    if mode == 'symmetric':
        Aff_knn = 0.5 * (C_knn + C_knn.T)
    elif mode == 'reciprocal':
        Aff_knn = C_knn.multiply(C_knn.T)
    else:
        raise Exception("Mode must be 'symmetric' or 'reciprocal'")
    return Aff_knn


def inference(loader, data_size, model, device):
    """
    :return:
    total_pred: prediction among all modalities
    pred_vectors: predictions of each modality, list
    labels_vector: true label
    Hs: high-level features
    Zs: low-level features
    """
    model.eval()
    soft_vector = []
    pred_vectors = []
    latent = []
    H = []
    for v in range(2):
        pred_vectors.append([])
        latent.append([])
        H.append([])
    labels_vector = []

    for batch_idx, (x0, x1, label) in enumerate(loader):
        x0, x1, label = x0.to(device), x1.to(device), label.to(device)
        with torch.no_grad():
            _, _, pred0, pred1 = model.forward_cluster(x0, x1)
            latent0, latent1, pseudo0, pseudo1, z0, z1 = model.forward(x0, x1)
            pseudo = (pseudo0 + pseudo1) / 2
            latent0 = latent0.detach()
            latent1 = latent1.detach()
            pred0 = pred0.detach()
            pred1 = pred1.detach()
            pred_vectors[0].extend(pred0.cpu().detach().numpy())
            pred_vectors[1].extend(pred1.cpu().detach().numpy())
            latent[0].extend(latent0.cpu().detach().numpy())
            latent[1].extend(latent1.cpu().detach().numpy())
        pseudo = pseudo.detach()
        soft_vector.extend(pseudo.cpu().detach().numpy())
        labels_vector.extend(label.cpu().detach().numpy())

    labels_vector = np.array(labels_vector).reshape(data_size)
    total_pred = np.argmax(np.array(soft_vector), axis=1)
    latent[0] = np.array(latent[0])
    latent[1] = np.array(latent[1])
    latent_fea = np.concatenate((latent[0], latent[1]), axis=1)
    pred_vectors[0] = np.array(pred_vectors[0])
    pred_vectors[1] = np.array(pred_vectors[1])

    return total_pred, pred_vectors, latent_fea, labels_vector


def valid(model, data_size, device, data, class_num, eval_h=False):
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=data_size,
        shuffle=True
    )
    total_pred, pred_vectors, high_level_vectors, labels_vector = inference(loader, data_size, model, device)
    view = 2
    if eval_h:
        logging.info("Clustering results on high-level features of each view:")

        kmeans = KMeans(n_clusters=class_num, n_init=100)
        y_pred = kmeans.fit_predict(high_level_vectors)
        nmi, ari, acc, pur = eva(labels_vector, y_pred)
        logging.info(
            "Clustering: acc={}, nmi={}, ari={}, PUR={}".format(acc,
                                                                nmi,
                                                                ari,
                                                                pur))

        logging.info("Clustering results on cluster assignments of each view:")
        for v in range(view):
            nmi, ari, acc, pur = eva(labels_vector, pred_vectors[v])
            logging.info(
                "Clustering: acc{}={}, nmi{}={}, ari{}={}, PUR{}={}".format(v + 1, acc,
                                                                            v + 1, nmi,
                                                                            v + 1, ari,
                                                                            v + 1, pur))

    logging.info("Clustering results on semantic labels: " + str(labels_vector.shape[0]))
    nmi, ari, acc, pur = eva(labels_vector, total_pred)
    logging.info(
        "Clustering: acc={}, nmi={}, ari={}, PUR={}".format(acc,
                                                            nmi,
                                                            ari,
                                                            pur))
    return acc, nmi, pur


def evaluate(C, data, labels, num_subspaces, spectral_dim, non_zeros=1000, n_neighbors=3,
             affinity='nearest_neighbor', knn_mode='symmetric'):
    C_sparse = get_sparse_rep(C=C, data=data, non_zeros=non_zeros)
    C_sparse_normalized = normalize(C_sparse).astype(np.float32)
    if affinity == 'symmetric':
        Aff = 0.5 * (np.abs(C_sparse_normalized) + np.abs(C_sparse_normalized).T)
    elif affinity == 'nearest_neighbor':
        Aff = get_knn_Aff(C_sparse_normalized, k=n_neighbors, mode=knn_mode)
    else:
        raise Exception("affinity should be 'symmetric' or 'nearest_neighbor'")
    preds = utils.spectral_clustering(Aff, num_subspaces, spectral_dim)
    acc = clustering_accuracy(labels, preds)
    nmi = normalized_mutual_info_score(labels, preds, average_method='geometric')
    ari = adjusted_rand_score(labels, preds)
    return acc, nmi, ari


def plot(acc, nmi, ari, args, data_name):
    x = range(0, args.con_epochs + 1, 1)
    fig_clustering = plt.figure()
    ax_clustering = fig_clustering.add_subplot(1, 1, 1)
    ax_clustering.set_title(data_name + ", " + "Noise=" + str(args.noisy_training) + ", RobustLoss=" + str(
        int(args.robust) * args.switching_time) + ", neg_prop=" + str(args.neg_prop))
    lns1 = ax_clustering.plot(x, acc, label='acc')
    lns2 = ax_clustering.plot(x, ari, label='ari')
    lns3 = ax_clustering.plot(x, nmi, label='nmi')
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax_clustering.legend(lns, labs, loc=0)
    ax_clustering.grid()
    ax_clustering.set_xlabel("epoch(s)")
    ax_clustering.plot()

    fig_dist = plt.figure()
    ax_dist_mean = fig_dist.add_subplot(1, 1, 1)
    ax_dist_mean.set_title(data_name + ", " + "Noise=" + str(args.noisy_training) + ", RobustLoss=" + str(
        int(args.robust) * args.switching_time) + ", neg_prop=" + str(args.neg_prop))
    lns1 = ax_dist_mean.plot(x, pos_dist_mean_list, label='pos. dist')
    lns2 = ax_dist_mean.plot(x, neg_dist_mean_list, label='neg. dist')
    lns3 = ax_dist_mean.plot(x, false_neg_dist_mean_list, label='false neg. dist')
    lns4 = ax_dist_mean.plot(x, true_neg_dist_mean_list, label='true neg. dist')
    lns = lns1 + lns2 + lns3 + lns4
    labs = [l.get_label() for l in lns]
    ax_dist_mean.legend(lns, labs, loc=0)
    ax_dist_mean.grid()
    ax_dist_mean.set_xlabel("epoch(s)")
    plt.show()


def main():  # deep features of Caltech101
    data_name = ['Scene15', 'Caltech101', 'Reuters_dim10', 'NoisyMNIST-30000', '2view-caltech101-8677sample',
                 'AWA-7view-10158sample', 'MNIST-USPS']

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    train_pair_loader, all_loader, divide_seed, class_num = loader(args.batch_size, args.neg_prop, args.aligned_prop,
                                                                   args.complete_prop, args.noisy_training,
                                                                   data_name[args.data])

    if args.data == 0:
        model = SUREfcScene().to(device)
        args.__setattr__('con_epochs', 80)
        args.__setattr__('lam1', 2.0)
        args.__setattr__('lam2', 0.1)
        args.__setattr__('edge', 0.5)
        args.__setattr__('temperature', 1.0)
        args.__setattr__('similarity_normalized', False)

    criterion_ncl = NoiseRobustLoss().to(device)
    criterion_mse = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate)
    if not os.path.exists('./log/'):
        os.mkdir("./log/")
    if not os.path.exists('./log/' + str(data_name[args.data]) + '/'):
        os.mkdir('./log/' + str(data_name[args.data]) + '/')
    path = os.path.join("./log/" + str(data_name[args.data]) + "/" + 'time=' + time
                        .strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    log_format = '%(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(path + '.txt')
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info("==========\nArgs:{}\n==========".format(args))
    logging.info("******** Training begin ********")

    train_time = 0
    epoch = 0
    LOSS = []
    LOSS_ioc = []
    LOSS_coc = []
    LOSS_rec = []
    while epoch < args.con_epochs:
        if epoch == 0:
            with torch.no_grad():
                pos_dist_mean, neg_dist_mean, false_neg_dist_mean, true_neg_dist_mean, \
                pos_cosine_mean, neg_cosine_mean, false_neg_cosine_mean, true_neg_cosine_mean, epoch_time, loss, loss_ioc, loss_coc, loss_rec = \
                    train(train_pair_loader, model, [criterion_ncl, criterion_mse], optimizer, epoch, args)
                LOSS.append(loss)
                LOSS_ioc.append(loss_ioc)
                LOSS_coc.append(loss_coc)
                LOSS_rec.append(loss_rec)
        else:
            pos_dist_mean, neg_dist_mean, false_neg_dist_mean, true_neg_dist_mean, \
            pos_cosine_mean, neg_cosine_mean, false_neg_cosine_mean, true_neg_cosine_mean, epoch_time, loss, loss_ioc, loss_coc, loss_rec = \
                train(train_pair_loader, model, [criterion_ncl, criterion_mse], optimizer, epoch, args)
            LOSS.append(loss)
            LOSS_ioc.append(loss_ioc)
            LOSS_coc.append(loss_coc)
            LOSS_rec.append(loss_rec)

            x0, x1, gt_label = both_infer(model, device, all_loader, args)
            x = [x0.T, x1.T]
            testdata = gettestdata(x, gt_label)
            acc, nmi, pur = valid(model, len(gt_label), device, testdata, class_num, eval_h=True)
        epoch += 1
        train_time += epoch_time

        pos_dist_mean_list.append(pos_dist_mean.item())
        neg_dist_mean_list.append(neg_dist_mean.item())
        true_neg_dist_mean_list.append(true_neg_dist_mean.item())
        false_neg_dist_mean_list.append(false_neg_dist_mean.item())

    logging.info('******** End, training time = {} s ********'.format(round(train_time, 2)))


if __name__ == '__main__':
    main()
