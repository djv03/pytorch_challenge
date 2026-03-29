import torch # type: ignore
import numpy as np
import os
import socket
from tqdm import tqdm
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm # type: ignore


hostname = socket.gethostname()

ROOT = '/home/jovyan/work/' if hostname == '7efc412e2797' else './'

class Trainer:
    def __init__(self,
                 model,                        
                 crit,
                 optim          = None,
                 train_dl       = None,
                 val_test_dl    = None,
                 cuda           = True,
                 early_stopping_patience = -1):  
        
        self._model         = model
        self._criterion     = crit
        self._optimizer     = optim
        self._train_dataloader  = train_dl
        self._val_dataloader    = val_test_dl
        self._use_cuda          = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model     = model.cuda()
            self._criterion = crit.cuda()
            
    def save_checkpoint(self, epoch):
        checkpoint_path = os.path.join(ROOT, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
        torch.save({'state_dict': self._model.state_dict()}, checkpoint_path)
    
    def restore_checkpoint(self, epoch_n):
        checkpoint_path = os.path.join(ROOT, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n))
        ckp = torch.load(checkpoint_path, 'cuda' if self._use_cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = torch.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        torch.onnx.export(m,             # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
            
    def train_step(self, x, y):
        self._optimizer.zero_grad()
        y_hat  = self._model(x)
        loss   = self._criterion(y_hat, y.float())
        loss.backward()
        self._optimizer.step()
        return loss.item()
        
        
    
    def val_test_step(self, x, y):
        output = self._model(x)
        loss   = self._criterion(output, y.float())
        output = output.detach().cpu().numpy()
        crack_prediction    = np.array(output[:, 0] > 0.5).astype(int)
        inactive_prediction = np.array(output[:, 1] > 0.5).astype(int)
        combined_prediction = np.stack([crack_prediction, inactive_prediction], axis = 1)
        return loss.item(), combined_prediction

    def train_epoch(self):
        
        self._model = self._model.train()
        mean_loss   = 0
        
        
        for x, y in tqdm(self._train_dataloader,desc="Training", unit="batch"):
            if self._use_cuda:
                x = x.cuda()
                y = y.cuda()
            loss = self.train_step(x, y)
            mean_loss += loss / len(self._train_dataloader)
        return mean_loss

    def val_test(self):
        self._model = self._model.eval()
        
        with torch.no_grad():
            mean_loss    = 0
            predictions  = []
            true_labels  = []

            for x, y in self._val_dataloader:
                if self._use_cuda:
                    x = x.cuda()
                    y = y.cuda()
                
                loss, pred = self.val_test_step(x, y)
                mean_loss += loss / len(self._val_dataloader)

                if self._use_cuda:
                    y = y.cpu()
                
                predictions.extend(pred)
                true_labels.extend(y.numpy())

            predictions, true_labels = np.array(predictions), np.array(true_labels)
            score = f1_score(true_labels, predictions, average = 'micro')
        
        return mean_loss, score
    
    def fit(self, epochs = -1):
        assert self._early_stopping_patience > 0 or epochs > 0
        train_losses = []
        val_losses   = []
        val_metrics  = []
        curr_epoch   = 0

        while curr_epoch < epochs:            
            train_loss = self.train_epoch()
            val_loss, val_metric = self.val_test()
            
            if len(val_losses) > 0 and val_loss < min(val_losses):
                self.save_checkpoint(curr_epoch)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_metrics.append(val_metric)

            if self._early_stopping_patience > 0 and len(val_losses) > self._early_stopping_patience:
                if all(val_losses[-i-1] > val_losses[-self._early_stopping_patience-1] for i in range(self._early_stopping_patience)):
                    break

            print(f'Epoch #{curr_epoch + 1:3d}: Training loss: {train_loss:.4f}  |  Validation loss: {val_loss:.4f}  |  F1 score: {val_metric:.4f}')
            curr_epoch += 1

        return train_losses, val_losses, val_metrics