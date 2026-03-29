import torch as t
import torch.nn as nn
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from model import ResNet
from trainer import Trainer

# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
# TODO
df= pd.read_csv('data.csv', sep=';')
train_df, val_df= train_test_split(df, test_size=0.2, random_state=42 )
print(train_df.head())

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
# TODO
train_dataset= ChallengeDataset(train_df,mode="train")
val_dataset= ChallengeDataset(val_df,mode="val")

BATCH_SIZE= 32

# train_loader= t.utils.data.DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)
# val_loader= t.utils.data.DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=2)
train_loader= t.utils.data.DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
val_loader= t.utils.data.DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=0)

# create an instance of our ResNet model
# TODO
model = ResNet()

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
criterion= nn.BCEWithLogitsLoss()

# set up the optimizer (see t.optim)
optimizer= t.optim.Adam(model.parameters(),lr=0.01)

# create an object of type Trainer and set its early stopping criterion
trainer= Trainer(
    model,                        # Model to be trained.
    crit=criterion,                         # Loss function
                 optim=optimizer,                   # Optimizer
                 train_dl=train_loader,                # Training data set
                 val_test_dl=val_loader,             # Validation (or test) data set
                 cuda=t.cuda.is_available(),                    # Whether to use the GPU
                 early_stopping_patience=-5,
)

# go, go, go... call fit on trainer
res= trainer.fit(epochs=2)

# plot the results
train_losses= res[0]
val_losses= res[1]
val_metrics= res[2]
epochs= len(train_losses)
x= np.arange(1,epochs+1)
plt.figure(figsize=(8,5))
plt.plot(x,train_losses,label="train loss",marker='o')
plt.plot(x, val_losses, label='Validation Loss', marker='x')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.grid(True)

# plt.plot(np.arange(len(res[0])), res[0], label='train loss')
# plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
# plt.legend()
plt.savefig('losses.png',dpi=300)
plt.show()