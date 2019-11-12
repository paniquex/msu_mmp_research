# Main file with train + test loops
# Question: Нужно ли делать два отдельных файла для трейн-цикла и инференс-цикла или же лучше все оставить тут?

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torchvision
# Logging
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

import pickle
import time
from tqdm import tqdm
# from fastprogress import master_bar, progress_bar



#my modules
import config
import dataset
import model
import metrics
import preprocessing


def train_model(fnames, x_train, y_train, train_transforms, conf):

    num_epochs = conf.num_epochs
    batch_size = conf.batch_size
    test_batch_size = conf.test_batch_size
    lr = conf.lr
    eta_min = conf.eta_min
    t_max = conf.t_max

    num_classes = y_train.shape[1]
    x_trn, x_val, y_trn, y_val = train_test_split(fnames,
                                                  y_train,
                                                  test_size=0.2,
                                                  random_state=conf.seed)

    train_dataset = dataset.TrainDataset(x_trn, None, y_trn, train_transforms)
    valid_dataset = dataset.TrainDataset(x_val, None, y_val, train_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              pin_memory=True,
                              num_workers=conf.n_jobs + 10,
                              worker_init_fn=conf.workers_init_fn
                              )
    valid_loader = DataLoader(valid_dataset, batch_size=test_batch_size, shuffle=False,
                              pin_memory=True,
                              num_workers=conf.n_jobs + 10,
                              worker_init_fn=conf.workers_init_fn
                              )

    net = model.MainModel(model_type='Simple', num_classes=num_classes)
    net = net.model.cuda()
    criterion = nn.BCEWithLogitsLoss().cuda()
    optimizer = optim.Adam(params=net.parameters(), lr=lr, amsgrad=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=t_max)

    best_epoch = -1
    best_lwlrap = 0.


    for epoch in range(num_epochs):
        start_time = time.time()
        net.train()
        avg_loss = 0.
        for x_batch, y_batch in train_loader:
            # grid = torchvision.utils.make_grid(x_batch)

            # conf.tb.add_graph(net, x_batch[0][0][0])
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
            preds = net(x_batch)
            loss = criterion(preds, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss += loss.item() / len(train_loader)
            torch.cuda.empty_cache()
        conf.tb.add_scalar('loss_train', avg_loss, epoch)

        net.eval()
        valid_preds = np.zeros((len(x_val), num_classes))
        avg_val_loss = 0.
        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(valid_loader):
                preds = net(x_batch.cuda()).detach()
                loss = criterion(preds, y_batch.cuda())

                preds = torch.sigmoid(preds)
                valid_preds[i * test_batch_size: (i + 1) * test_batch_size] = preds.cpu().numpy()

                avg_val_loss += loss.item() / len(valid_loader)
            conf.tb.add_scalar('loss_val', avg_val_loss, epoch)

        score, weight = metrics.calculate_per_class_lwlrap(y_val, valid_preds)
        lwlrap = (score * weight).sum()

        scheduler.step()
        # scheduler.step(avg_val_loss)
        if (epoch + 1) % 5 == 0:
            elapsed = time.time() - start_time
            print(
                f'Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  val_lwlrap: {lwlrap:.6f}  time: {elapsed:.0f}s')

        conf.tb.add_scalar('val_lwlrap', lwlrap, epoch)

        if lwlrap > best_lwlrap:
            best_epoch = epoch + 1
            best_lwlrap = lwlrap
            torch.save(net.state_dict(), 'weight_best.pt')

    conf.tb.close()
    return {
        'best_epoch': best_epoch,
        'best_lwlrap': best_lwlrap,
    }


def predict_model(test_fnames, x_test, test_transforms, num_classes, *, tta=5):
    batch_size = 64

    test_dataset = dataset.TestDataset(test_fnames, x_test, test_transforms, tta=tta)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    net = model.MainModel(model_type='Simple', num_classes=num_classes)
    net = net.model.cuda()
    net.load_state_dict(torch.load('weight_best.pt'))
    net.cuda()
    net.eval()

    all_outputs, all_fnames = [], []

    for images, fnames in test_loader:
        preds = torch.sigmoid(net(images.cuda()).detach())
        all_outputs.append(preds.cpu().numpy())
        all_fnames.extend(fnames)

    test_preds = pd.DataFrame(data=np.concatenate(all_outputs),
                              index=all_fnames,
                              columns=map(str, range(num_classes)
                                          )
                              )
    test_preds = test_preds.groupby(level=0).mean()

    return test_preds


def main():
    conf = config.Config(80) # conf.num_classes = 80
    conf.seed_torch()
    train_curated = pd.read_csv(conf.csv_file['train_curated'])
    train_noisy = pd.read_csv(conf.csv_file['train_noisy'])
    train_df = pd.concat([train_curated], sort=True, ignore_index=True)

    test_df = pd.read_csv(conf.csv_file['sample_submission'])
    print('CSV files loading was done.')

    labels = test_df.columns[1:].tolist()
    y_train = np.zeros((len(train_df), conf.num_classes)).astype(int)
    for i, row in enumerate(train_df['labels'].str.split(',')):
        for label in row:
            idx = labels.index(label)
            y_train[i, idx] = 1

    print(y_train.shape)

    ## loading preprocessed melspectrograms
    with open(conf.mels['train_curated'], 'rb') as curated, open(conf.mels['train_noisy'], 'rb') as noisy:
        x_train = pickle.load(curated)
        x_train.extend(pickle.load(noisy))

    with open(conf.mels['test'], 'rb') as test:
        x_test = pickle.load(test)

    # augmentations
    transforms_dict = {
        'train': transforms.Compose([
            transforms.RandomCrop(128),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.RandomCrop(128),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
        ]),
    }
    print(len(x_train), len(x_test))

    #logging
    t_max_list = [conf.t_max]
    lr_list = [conf.lr]
    batch_size_list = [conf.batch_size]
    epoch_amount_list = [conf.num_epochs]
    for t_max in t_max_list:
        for batch_size in batch_size_list:
            for lr in lr_list:
                for epoch in epoch_amount_list:
                    conf.t_max = t_max
                    conf.batch_size = batch_size
                    conf.lr = lr
                    conf.num_epochs = epoch
                    comment = f'_batch_size={batch_size}_lr={lr}_t_max={t_max}_epoches={epoch}_scheduler=COSINE_preproc={conf.preprocessing_type}'
                    conf.tb = SummaryWriter(comment=comment)
                    result = train_model(train_df['fname'], None, y_train, transforms_dict['train'], conf=conf)
                    print(result)
    test_preds = predict_model(test_df['fname'], None, transforms_dict['test'], conf.num_classes, tta=20)
    test_df[labels] = test_preds.values
    test_df.to_csv('submission.csv', index=False)
    test_df.head()


if __name__ == '__main__':
    main()
