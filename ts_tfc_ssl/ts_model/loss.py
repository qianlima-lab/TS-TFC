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
    anchor_dot_contrast = torch.div(
        torch.matmul(embd_batch, embd_batch.T),
        temperature)

    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)

    logits_mask = torch.scatter(
        torch.ones_like(logits.detach()),
        1,
        torch.arange(embd_batch.shape[0]).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask
    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
    num_anchor = 0
    for s in mask.sum(1):
        if s != 0:
            num_anchor = num_anchor + 1
    # loss
    loss = - (temperature / base_temperature) * mean_log_prob_pos
    loss = loss.sum(0) / (num_anchor + 1e-12)

    return loss
