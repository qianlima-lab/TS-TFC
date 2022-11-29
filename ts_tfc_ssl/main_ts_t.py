import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import argparse
import queue
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from ts_tfc_ssl.ts_data.dataloader import UCRDataset
from ts_tfc_ssl.ts_data.preprocessing import normalize_per_series, fill_nan_value
from ts_tfc_ssl.ts_model.loss import sup_contrastive_loss
from ts_tfc_ssl.ts_model.model import ProjectionHead
from ts_tfc_ssl.ts_utils import build_model, set_seed, build_dataset, get_all_datasets, \
    construct_graph_via_knn_cpl_nearind_gpu, \
    build_loss, shuffler, evaluate

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Base setup
    parser.add_argument('--backbone', type=str, default='fcn', help='encoder backbone, fcn')
    parser.add_argument('--random_seed', type=int, default=42, help='uniform random seed')

    # Dataset setup
    parser.add_argument('--dataset', type=str, default='GunPoint',
                        help='dataset(in ucr)')  #  GunPointOldVersusYoung GunPoint
    parser.add_argument('--dataroot', type=str, default='./UCRArchive_2018', help='path of UCR folder')
    parser.add_argument('--num_classes', type=int, default=0, help='number of class')
    parser.add_argument('--input_size', type=int, default=1, help='input_size')

    # Semi training
    parser.add_argument('--labeled_ratio', type=float, default=0.1, help='0.1, 0.2, 0.4')
    parser.add_argument('--warmup_epochs', type=int, default=300, help='warmup epochs using only labeled data for ssl')
    parser.add_argument('--queue_maxsize', type=int, default=3, help='2 or 3')
    parser.add_argument('--knn_num', type=int, default=5, help='5 or 10')

    # Contrastive loss
    parser.add_argument('--sup_con_mu', type=float, default=0.05, help='0.005 or 0.05')
    parser.add_argument('--mlp_head', type=bool, default=True, help='mlp head project')
    parser.add_argument('--temperature', type=float, default=50, help='20 or 50')

    # training setup
    parser.add_argument('--loss', type=str, default='cross_entropy', help='loss function')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=1024, help='')
    parser.add_argument('--epoch', type=int, default=1000, help='training epoch')
    parser.add_argument('--cuda', type=str, default='cuda:0')

    # classifier setup
    parser.add_argument('--classifier', type=str, default='linear', help='')
    parser.add_argument('--classifier_input', type=int, default=128, help='input dim of the classifiers')

    args = parser.parse_args()

    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    set_seed(args)

    sum_dataset, sum_target, num_classes = build_dataset(args)
    args.num_classes = num_classes

    while sum_dataset.shape[0] * 0.6 < args.batch_size:
        args.batch_size = args.batch_size // 2

    if args.batch_size * 2 > sum_dataset.shape[0] * 0.6:
        args.queue_maxsize = 2

    model, classifier = build_model(args)
    projection_head = ProjectionHead(input_dim=128)
    model, classifier = model.to(device), classifier.to(device)
    projection_head = projection_head.to(device)
    loss = build_loss(args).to(device)

    model_init_state = model.state_dict()
    classifier_init_state = classifier.state_dict()
    projection_head_init_state = projection_head.state_dict()

    is_projection_head = args.mlp_head
    optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': classifier.parameters()},
                                  {'params': projection_head.parameters()}], lr=args.lr)

    print('start ssl on {}'.format(args.dataset))

    train_datasets, train_targets, val_datasets, val_targets, test_datasets, test_targets = get_all_datasets(
        sum_dataset, sum_target)

    losses = []
    test_accuracies = []
    train_time = 0.0
    end_val_epochs = []

    for i, train_dataset in enumerate(train_datasets):
        t = time.time()

        model.load_state_dict(model_init_state)
        classifier.load_state_dict(classifier_init_state)
        projection_head.load_state_dict(projection_head_init_state)
        print('{} fold start training and evaluate'.format(i))

        train_target = train_targets[i]
        val_dataset = val_datasets[i]
        val_target = val_targets[i]

        test_dataset = test_datasets[i]
        test_target = test_targets[i]

        train_dataset, val_dataset, test_dataset = fill_nan_value(train_dataset, val_dataset, test_dataset)

        # TODO normalize per series
        train_dataset = normalize_per_series(train_dataset)
        val_dataset = normalize_per_series(val_dataset)
        test_dataset = normalize_per_series(test_dataset)

        train_labeled, train_unlabeled, y_labeled, y_unlabeled = train_test_split(train_dataset, train_target,
                                                                                  test_size=(1 - args.labeled_ratio),
                                                                                  random_state=args.random_seed)

        mask_labeled = np.zeros(len(y_labeled))
        mask_unlabeled = np.ones(len(y_unlabeled))
        mask_train = np.concatenate([mask_labeled, mask_unlabeled])
        train_all_split = np.concatenate([train_labeled, train_unlabeled])
        y_label_split = np.concatenate([y_labeled, y_unlabeled])

        x_train_all, y_train_all = shuffler(train_all_split, y_label_split)
        mask_train, _ = shuffler(mask_train, mask_train)
        y_train_all[mask_train == 1] = -1  ## Generate unlabeled data

        x_train_all = torch.from_numpy(x_train_all).to(device)
        y_train_all = torch.from_numpy(y_train_all).to(device).to(torch.int64)

        x_train_labeled_all = x_train_all[mask_train == 0]
        y_train_labeled_all = y_train_all[mask_train == 0]

        train_set_labled = UCRDataset(x_train_labeled_all, y_train_labeled_all)
        train_set = UCRDataset(x_train_all, y_train_all)
        val_set = UCRDataset(torch.from_numpy(val_dataset).to(device),
                             torch.from_numpy(val_target).to(device).to(torch.int64))
        test_set = UCRDataset(torch.from_numpy(test_dataset).to(device),
                              torch.from_numpy(test_target).to(device).to(torch.int64))

        batch_size_labeled = 128  ## warmup batch
        while x_train_labeled_all.shape[0] < batch_size_labeled:
            batch_size_labeled = batch_size_labeled // 2

        if x_train_labeled_all.shape[0] < 16:
            batch_size_labeled = 16

        train_labeled_loader = DataLoader(train_set_labled, batch_size=batch_size_labeled, num_workers=0,
                                          drop_last=False)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=0, drop_last=False)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=0)

        last_loss = float('inf')
        stop_count = 0
        increase_count = 0

        num_steps = train_set.__len__() // args.batch_size

        min_val_loss = float('inf')
        test_accuracy = 0
        end_val_epoch = 0

        queue_train_x = queue.Queue(args.queue_maxsize)
        queue_train_y = queue.Queue(args.queue_maxsize)
        queue_train_mask = queue.Queue(args.queue_maxsize)

        for epoch in range(args.epoch):

            if stop_count == 80 or increase_count == 80:
                print('model convergent at epoch {}, early stopping'.format(epoch))
                break

            num_iterations = 0

            model.train()
            classifier.train()
            projection_head.train()

            if epoch < args.warmup_epochs:
                for x, y in train_labeled_loader:

                    # print("x.shape = ", x.shape, ", y.shape = ", y.shape)

                    if x.shape[0] < 2:
                        continue
                    optimizer.zero_grad()
                    pred_embed = model(x)
                    if is_projection_head:
                        preject_head_embed = projection_head(pred_embed)

                    pred = classifier(pred_embed)
                    step_loss = loss(pred, y)

                    if len(y) > 1:
                        batch_sup_contrastive_loss = sup_contrastive_loss(
                            embd_batch=preject_head_embed,
                            labels=y,
                            device=device,
                            temperature=args.temperature,
                            base_temperature=args.temperature)

                        step_loss = step_loss + batch_sup_contrastive_loss * args.sup_con_mu

                    step_loss.backward()
                    optimizer.step()

            else:
                for x, y in train_loader:
                    if x.shape[0] < 2:
                        continue
                    optimizer.zero_grad()
                    pred_embed = model(x)
                    if is_projection_head:
                        preject_head_embed = projection_head(pred_embed)

                    if (num_iterations + 1) * args.batch_size < train_set.__len__():
                        mask_train_batch = mask_train[
                                           num_iterations * args.batch_size: (num_iterations + 1) * args.batch_size]
                    else:
                        mask_train_batch = mask_train[num_iterations * args.batch_size:]

                    mask_cpl_batch = torch.tensor([False] * len(mask_train_batch)).to(device)

                    if epoch >= args.warmup_epochs:
                        if not queue_train_x.full():
                            queue_train_x.put(preject_head_embed.detach())
                            queue_train_y.put(y)
                            queue_train_mask.put(mask_train_batch)

                        if queue_train_x.full():
                            train_x_allq = queue_train_x.queue
                            train_y_allq = queue_train_y.queue
                            train_mask_allq = queue_train_mask.queue

                            embed_train_x_allq = torch.cat([train_x_allq[j] for j in range(len(train_x_allq))], 0)
                            y_label_allq = torch.cat([train_y_allq[j] for j in range(len(train_y_allq))], 0)
                            mask_lable_allq = np.concatenate(train_mask_allq)

                            _, end_knn_label, mask_cpl_knn, _ = construct_graph_via_knn_cpl_nearind_gpu(
                                data_embed=embed_train_x_allq, y_label=y_label_allq,
                                mask_label=mask_lable_allq, device=device,
                                num_real_class=args.num_classes, topk=args.knn_num)
                            knn_result_label = torch.tensor(end_knn_label).to(device).to(torch.int64)

                            y[mask_train_batch == 1] = knn_result_label[(len(knn_result_label) - len(y)):][
                                mask_train_batch == 1]
                            mask_cpl_batch[mask_train_batch == 1] = mask_cpl_knn[(len(mask_cpl_knn) - len(y)):][
                                mask_train_batch == 1]

                            _ = queue_train_x.get()
                            _ = queue_train_y.get()
                            _ = queue_train_mask.get()

                    mask_clean = [True if mask_train_batch[m] == 0 else False for m in range(len(mask_train_batch))]
                    mask_select_loss = [False for m in range(len(y))]
                    for m in range(len(mask_train_batch)):
                        if mask_train_batch[m] == 0:
                            mask_select_loss[m] = True
                        else:
                            if mask_cpl_batch[m]:
                                mask_select_loss[m] = True

                    pred = classifier(pred_embed)
                    step_loss = loss(pred[mask_select_loss], y[mask_select_loss])

                    if epoch > args.warmup_epochs:

                        if len(y[mask_train_batch == 0]) > 1:
                            batch_sup_contrastive_loss = sup_contrastive_loss(
                                embd_batch=preject_head_embed[mask_train_batch == 0],
                                labels=y[mask_train_batch == 0],
                                device=device,
                                temperature=args.temperature,
                                base_temperature=args.temperature)

                            step_loss = step_loss + batch_sup_contrastive_loss * args.sup_con_mu

                    step_loss.backward()
                    optimizer.step()

                    num_iterations += 1

            model.eval()
            classifier.eval()
            projection_head.eval()

            val_loss, val_accu = evaluate(val_loader, model, classifier, loss, device)
            if min_val_loss > val_loss:
                min_val_loss = val_loss
                end_val_epoch = epoch
                test_loss, test_accuracy = evaluate(test_loader, model, classifier, loss, device)

            if (epoch > args.warmup_epochs) and (abs(last_loss - val_loss) <= 1e-4):
                stop_count += 1
            else:
                stop_count = 0

            if (epoch > args.warmup_epochs) and (val_loss > last_loss):
                increase_count += 1
            else:
                increase_count = 0

            last_loss = val_loss

            if epoch % 50 == 0:
                print("epoch : {},  test_accuracy : {}".format(epoch, test_accuracy))

        test_accuracies.append(test_accuracy)
        end_val_epochs.append(end_val_epoch)
        t = time.time() - t
        train_time += t

        print('{} fold finish training'.format(i))

    test_accuracies = torch.Tensor(test_accuracies)
    end_val_epochs = np.array(end_val_epochs)

    print("Training end: mean_test_acc = ", round(torch.mean(test_accuracies).item(), 4), "traning time (seconds) = ",
          round(train_time, 4), ", seed = ", args.random_seed)
    print('Done!')
