import torch
import torch.nn as nn

EPS = 1e-8


def cross_entropy():
    loss = nn.CrossEntropyLoss()
    return loss


def reconstruction_loss():
    loss = nn.MSELoss()
    return loss


def sup_contrastive_loss(embd_batch, labels, device,
                         temperature=0.07, base_temperature=0.07):
    # embd_batch = torch.nn.functional.normalize(embd_batch, dim=1)
    # p = F.normalize(p, dim=-1) ##

    anchor_dot_contrast = torch.div(
        torch.matmul(embd_batch, embd_batch.T),
        temperature)

    # print("anchor_dot_contrast.shape = ", anchor_dot_contrast.shape)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # print("labels = ", labels)
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)
    # print("mask = ", mask)
    # mask_clean = torch.tensor(mask_clean).to(device)
    # mask = mask_clean.int()
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(logits.detach()),
        1,
        torch.arange(embd_batch.shape[0]).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask
    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask

    # print("exp_logits.shape = ", exp_logits.shape, torch.log(exp_logits.sum(1, keepdim=True) + 1e-12).shape)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
    # log_prob =
    # print("log_prob.shape = ", log_prob.shape)
    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

    num_anchor = 0
    for s in mask.sum(1):
        if s != 0:
            num_anchor = num_anchor + 1
    # loss
    loss = - (temperature / base_temperature) * mean_log_prob_pos
    # print("len_loss = ", len(loss), loss.shape, mask.sum(1), loss)
    # print("sum_loss = ", loss.sum(0), loss.mean() * loss.shape[0], ", num_anchor = ", num_anchor)
    # loss = loss.mean()
    loss = loss.sum(0) / (num_anchor + 1e-12)

    return loss
