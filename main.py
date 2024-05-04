from torch_snippets import *
import torch
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from xml.etree import ElementTree as et
import numpy as np
import wandb
import math
import FruitsDataset
import modelConfig

# Global settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Paths
root = 'fruit-images-for-object-detection/train_zip/train/'
val_root = 'fruit-images-for-object-detection/test_zip/test'
checkpoint = 'checkpoints/FRCNN.pth'

# we have four labels
labels = ['background', 'orange', 'apple', 'banana']
label2targets = {l: t for t, l in enumerate(labels)}
targets2label = {t: l for l, t in label2targets.items()}
num_classes = len(targets2label)

# Hyperparameters
batch_size_train = 4
batch_size_val = 2
learning_rate = 1e-5
weight_decay = 1e-3
momentum =0.9
optimizer_pars = {'lr': learning_rate, 'weight_decay': weight_decay, 'momentum': momentum }


print("label2targets",label2targets)
print("targets2label",targets2label)





tr_ds = FruitsDataset.FruitsDataset()
tr_dl = DataLoader(tr_ds, batch_size=batch_size_train, shuffle=True, collate_fn=tr_ds.collate_fn)

val_ds = FruitsDataset.FruitsDataset(root=val_root)
val_dl = DataLoader(val_ds, batch_size=batch_size_val, shuffle=True, collate_fn=val_ds.collate_fn)


''''''
# test the model
imgs, targets = next(iter(tr_dl))
imgs = list(img.to(device) for img in imgs)
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
model = modelConfig.get_model(num_classes).to(device)
model(imgs, targets)


# Create Model
model = modelConfig.get_model(num_classes).to(device)
# optim = torch.optim.Adam(model.parameters(), **optimizer_pars)
optim = torch.optim.SGD(model.parameters(), **optimizer_pars)

log_name = f"FRCNN_V1_SGD_lr_{optimizer_pars['lr']}_weight_decay_{optimizer_pars['weight_decay']}_momentum_{optimizer_pars['momentum']}"
#wandb.login()
#wandb.init(project='INM706_CW', name=log_name )


def train_batch(batch, model, optim):
    model.train()
    imgs, targets = batch
    imgs = list(img.to(device) for img in imgs)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    optim.zero_grad()
    losses = model(imgs, targets)
    loss = sum(loss for loss in losses.values())
    loss.backward()
    optim.step()
    return loss, losses

@torch.no_grad()
def validate_batch(batch, model, optim):
    model.train()
    imgs, targets = batch
    imgs = list(img.to(device) for img in imgs)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    optim.zero_grad()
    losses = model(imgs, targets)
    loss = sum(loss for loss in losses.values())
    return loss, losses







##################################
# TRAIN


n_epochs = 5
min_loss = 100
epoch_loss = 0
log = Report(n_epochs)

for e in range(n_epochs):
    for i, batch in enumerate(tr_dl):
        N = len(tr_dl)
        loss, losses = train_batch(batch, model, optim)
        loc_loss, regr_loss, loss_objectness, loss_rpn_box_reg = [losses[k] for k in
                                                                  ['loss_classifier', 'loss_box_reg', 'loss_objectness',
                                                                   'loss_rpn_box_reg']]
        log.record(e + (i + 1) / N, trn_loss=loss.item(), trn_loc_loss=loc_loss.item(),
                   trn_regr_loss=regr_loss.item(), trn_loss_objectness=loss_objectness.item(),
                   trn_loss_rpn_box_reg=loss_rpn_box_reg.item())
        # Log metrics with wandb
        '''wandb.log({
            'trn_loss': loss,
            'trn_loc_loss': loc_loss,
            'trn_regr_loss': regr_loss,
            'trn_loss_objectness': loss_objectness,
            'trn_loss_rpn_box_reg': loss_rpn_box_reg
        })'''
        epoch_loss += loss.item()

    epoch_loss /= len(tr_dl)

    if epoch_loss < min_loss:
        torch.save(model.state_dict(), 'fruit-images-for-object-detection/checkpoints/FRCNN.pth')

    #wandb.log({'train_loss': epoch_loss})
    print("Loss={0:.4f} in Epoch={1:d}".format(epoch_loss, e))

    log.report_avgs(e + 1)

for e in range(n_epochs):
    # Validation loop
    for i, batch in enumerate(val_dl):
        N = len(val_dl)
        loss, losses = validate_batch(batch, model.float(), optim)
        loc_loss, regr_loss, loss_objectness, loss_rpn_box_reg = [losses[k] for k in
                                                                  ['loss_classifier', 'loss_box_reg',
                                                                   'loss_objectness', 'loss_rpn_box_reg']]
        log.record(e + (i + 1) / N, val_loss=loss.item(), val_loc_loss=loc_loss.item(),
                   val_regr_loss=regr_loss.item(), val_loss_objectness=loss_objectness.item(),
                   val_loss_rpn_box_reg=loss_rpn_box_reg.item())
        # Log metrics with wandb
        '''wandb.log({
            'val_loss': loss,
            'val_loc_loss': loc_loss,
            'val_regr_loss': regr_loss,
            'val_loss_objectness': loss_objectness,
            'val_loss_rpn_box_reg': loss_rpn_box_reg
        })'''

    log.report_avgs(e + 1)

log.plot_epochs(['trn_loss', 'val_loss'])
#wandb.finish()






def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
