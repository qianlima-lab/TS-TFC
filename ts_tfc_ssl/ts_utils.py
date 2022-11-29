import random
from collections import Counter

import numpy as np
import torch
import torch.optim

from ts_tfc_ssl.ts_data.preprocessing import load_data, transfer_labels, k_fold
from ts_tfc_ssl.ts_model.loss import cross_entropy, reconstruction_loss
from ts_tfc_ssl.ts_model.model import FCN, Classifier


def set_seed(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)


def build_model(args):
    if args.backbone == 'fcn':
        model = FCN(args.num_classes, args.input_size)

    if args.classifier == 'linear':
        classifier = Classifier(args.classifier_input, args.num_classes)

    return model, classifier


def build_dataset(args):
    sum_dataset, sum_target, num_classes = load_data(args.dataroot, args.dataset)

    sum_target = transfer_labels(sum_target)
    return sum_dataset, sum_target, num_classes


def build_loss(args):
    if args.loss == 'cross_entropy':
        return cross_entropy()
    elif args.loss == 'reconstruction':
        return reconstruction_loss()


def shuffler(x_train, y_train):
    indexes = np.array(list(range(x_train.shape[0])))
    np.random.shuffle(indexes)
    x_train = x_train[indexes]
    y_train = y_train[indexes]
    return x_train, y_train


def get_all_datasets(data, target):
    return k_fold(data, target)


def convert_coeff(x, eps=1e-6):
    amp = torch.sqrt((x.real + eps).pow(2) + (x.imag + eps).pow(2))
    phase = torch.atan2(x.imag, x.real + eps)
    stack_r = torch.stack((amp, phase), -1)
    stack_r = stack_r.permute(0, 2, 1)
    return stack_r, phase


def evaluate(val_loader, model, classifier, loss, device):
    val_loss = 0
    val_accu = 0

    sum_len = 0
    for data, target in val_loader:
        '''
        data, target = data.to(device), target.to(device)
        target = target.to(torch.int64)
        '''
        with torch.no_grad():
            val_pred = model(data)
            val_pred = classifier(val_pred)
            val_loss += loss(val_pred, target).item()

            val_accu += torch.sum(torch.argmax(val_pred.data, axis=1) == target)
            sum_len += len(target)

    return val_loss / sum_len, val_accu / sum_len


def construct_graph_via_knn_cpl_nearind_gpu(data_embed, y_label, mask_label, device, topk=5, sigma=0.25, alpha=0.99,
                                            p_cutoff=0.95, num_real_class=2):
    eps = np.finfo(float).eps
    n, d = data_embed.shape[0], data_embed.shape[1]
    data_embed = data_embed
    emb_all = data_embed / (sigma + eps)  # n*d
    emb1 = torch.unsqueeze(emb_all, 1)  # n*1*d
    emb2 = torch.unsqueeze(emb_all, 0)  # 1*n*d
    w = ((emb1 - emb2) ** 2).mean(2)  # n*n*d -> n*n
    w = torch.exp(-w / 2)

    ## keep top-k values
    topk, indices = torch.topk(w, topk)
    mask = torch.zeros_like(w).to(device)
    mask = mask.scatter(1, indices, 1)
    mask = ((mask + torch.t(mask)) > 0).type(torch.float32)  # union, knn graph
    # mask = ((mask>0)&(torch.t(mask)>0)).type(torch.float32)  # intersection, knn graph
    w = w * mask

    ## normalize
    d = w.sum(0)
    d_sqrt_inv = torch.sqrt(1.0 / (d + eps)).to(device)
    d1 = torch.unsqueeze(d_sqrt_inv, 1).repeat(1, n)
    d2 = torch.unsqueeze(d_sqrt_inv, 0).repeat(n, 1)
    s = d1 * w * d2

    # step2: label propagation, f = (i-\alpha s)^{-1}y
    y = torch.zeros(y_label.shape[0], num_real_class)
    for i in range(len(mask_label)):
        if mask_label[i] == 0:
            y[i][int(y_label[i])] = 1
    f = torch.matmul(torch.inverse(torch.eye(n).to(device) - alpha * s + eps), y.to(device))
    all_knn_label = torch.argmax(f, 1).cpu().numpy()
    end_knn_label = torch.argmax(f, 1).cpu().numpy()

    class_counter = Counter(y_label)
    for i in range(num_real_class):
        class_counter[i] = 0
    for i in range(len(mask_label)):
        if mask_label[i] == 0:
            end_knn_label[i] = y_label[i]
        else:
            class_counter[all_knn_label[i]] += 1

    classwise_acc = torch.zeros((num_real_class,)).to(device)
    for i in range(num_real_class):
        classwise_acc[i] = class_counter[i] / max(class_counter.values())
    pseudo_label = torch.softmax(f, dim=-1)
    max_probs, max_idx = torch.max(pseudo_label, dim=-1)
    cpl_mask = max_probs.ge(p_cutoff * (classwise_acc[max_idx] / (2. - classwise_acc[max_idx])))

    return all_knn_label, end_knn_label, cpl_mask, indices
