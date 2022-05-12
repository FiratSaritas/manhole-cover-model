import os
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import KFold
import numpy as np
from tqdm.notebook import tqdm
import torch.nn.functional as F
import wandb
import PIL
import pandas as pd

class CheckPoint:
    
    def __init__(self, history, history_folder, name):
        self.history = history
        self.history_folder = history_folder + '_' + name    
        
    def _init_history(self):
        if os.path.isdir(self.history_folder):
            print(f'Found Existing History Folder. Removing all Files: {self.history_folder}')
            all_files_in_hist = os.listdir(self.history_folder)
            for file in all_files_in_hist:
                os.remove(os.path.join(self.history_folder, file))
        else:
            print(f'No History Folder. Create History folder at: {self.history_folder}')
            os.mkdir(self.history_folder)
                
    def load_from_history(self, epoch: int, inplace: bool = False):
        """
        Loads Network from history given the Epoch
        """
        assert self.history == True, 'No history'
        model = torch.load(os.path.join(self.history_folder,  f'{self.name}_{epoch}'))
        if inplace:
            self.model = torch.load(os.path.join(self.history_folder,  f'{self.name}_{epoch}'))
            print(f'Replaced self.model -> epoch {epoch}')
        else:
            return model


class TrainingInterface(CheckPoint):
    """
    This class implements a simple Sklearn like interface for training of Neural Network
    """
    
    def __init__(self, model, name, history: bool = False, history_folder: str = 'train_history', writer: object = None):
        """
        Training Interface Wrapper class for the training of neural network classifier
        in pytorch. Only applicable for image classification.
        
        Params:
        -------------------
        model: (torch.model)     Neural Network class Pytorch     
        name: (str)              Name of Neural Network
        history: (bool)          If true saves trained model in history to restore best epochs 
                                 at end of training.
        history_folder: (bool)   Folder where the train history is saved. 
        writer: (object)         If true uses wandb to log all outputs during training and inference
        
        dev:                     Device Cuda or cpu
        train_losses:            Training losses recorded during training
        eval_losses:             Validation Losses recorded during training
        """
        super().__init__(history=history, history_folder=history_folder, name=name)
        self.model = model
        self.name = name 
        if self.history:
            self._init_history()
        self.writer = writer
        
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.epoch = 0
        self.train_loss = []
        self.val_loss = []
        self.train_epoch_loss = []
        self.val_epoch_loss = []
        

    def print_network(self):
        """
        Prints networks and its layers.
        """
        print(self.model)
    
    def print_total_params(self, return_=False):
        """
        Prints total params.
        
        Params:
        -------------      
        return_:            if return the the result will be returned        
        """
        pytorch_total_params = sum(p.numel() for p in self.model.parameters())
                                  
        if not return_:
            print(50 * '=')
            print(f'{self.name} | Trainable Parameters: {pytorch_total_params}')
            print(50 * '=')
        else:
            return '{}\n{}\n{}'.format(50 * '=', pytorch_total_params, 50 * '=')
        
    def train(self, criterion, optimizer, n_epochs, dataloader_train, 
                      dataloader_val=None, epsilon=.0001, verbose=True):
        """
        Trains a neural Network with given inputs and parameters.

        params:
        -----------------
        model:                Neural Network class Pytorch     
        criterion:            Cost-Function used for the network optimizatio
        optimizer:            Optmizer for the network
        n_epochs:             Defines how many times the whole dateset should be fed through the network
        dataloader_train:     Dataloader with the batched dataset
        dataloader_val:       Dataloader test with the batched dataset used for test loop. If None -> No eval loop
        epsilon:              Epsilon defines stop criterion regarding to convergence of loss
        verbose:               Prints Report after each Epoch
        
        """
        y_pred, y_true = torch.Tensor(), torch.Tensor()
        self.model.to(self.dev)
        criterion.to(self.dev)

        self.model.train()
        overall_length = len(dataloader_train)
        with tqdm(total=n_epochs*overall_length) as pbar:
            for epoch in range(n_epochs):  # loop over the dataset multiple times
                running_loss, val_loss = 0., 0.
                for i, data in enumerate(dataloader_train):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data
                    inputs, labels = inputs.to(self.dev), labels.to(self.dev)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # calc and print stats
                    self.train_loss.append(loss.item())
                    if self.writer != None:
                        self.writer.log({'train_batch_loss': loss.item()})
                        
                    running_loss += loss.item()                
                    pbar.set_description('Epoch: {}/{} // Running Loss: {} '.format(epoch+1, n_epochs, 
                                                                                    np.round(running_loss, 3)))   
                    pbar.update(1)
                
                self.train_epoch_loss.append(running_loss)
                if self.writer != None:
                    self.writer.log({'train_epoch_loss': running_loss})

                if dataloader_val:
                    length_dataloader_val = len(dataloader_val)
                    val_loss = 0.
                    for i, data in enumerate(dataloader_val):
                        pbar.set_description(f'Epoch: {epoch+1}/{n_epochs} // Eval-Loop: {i+1}/{length_dataloader_val}')
                        self.model.eval()
                        # get the inputs; data is a list of [inputs, labels]
                        inputs, labels = data
                        inputs, labels = inputs.to(self.dev), labels.to(self.dev)
                        with torch.no_grad():
                            outputs = self.model(inputs)
                            eval_loss = criterion(outputs, labels)
                            val_loss += eval_loss.item()
                            self.val_loss.append(eval_loss.item())
                            if self.writer != None:
                                self.writer.log({'val_batch_loss': eval_loss.item()})
                        self.model.train()  
                    self.val_epoch_loss.append(val_loss)
                    
                    if self.writer != None:
                        self.writer.log({'val_epoch_loss': val_loss})
                
                # Update epoch
                self.epoch += 1
                
                if verbose:
                    print('Epoch {}/{}: [Train-Loss = {}] || [Validation-Loss = {}]'.format(self.epoch, n_epochs, 
                                                                                         np.round(running_loss, 3),     
                                                                                         np.round(val_loss, 3)))     
                if epoch > 0:
                    if epsilon > np.abs(loss_before - running_loss):
                        print(20*'=', 'Network Converged', 20*'=')
                        break
                loss_before = running_loss

                if self.history:
                    torch.save(self.model, 
                               os.path.join(self.history_folder,  f'{self.name}_{self.epoch}'))
                    
        return self
    
    def predict(self, dataloader, return_images: bool = True, return_prob: bool = True, 
                disable_pbar: bool = False):
        """
        Returns true and predicted labels for prediction
        Params:
        ---------
        model:           Pytorch Neuronal Net
        dataloader:      batched Testset
        return_images:   If true returns images
        return_prob:     If true returns predicted probabilities
        disable_pbar:    If true disables pbar
        returns:
        ----------
        (y_true, y_pred, y_images, y_prob): 
            y_true       True labels
            y_pred:      Predicted Labels
            y_prob:      Predicted Probability (empty if return_prob = False)
            y_images:    Images (empty if return_images = False)
        """
        self.model.to(self.dev)
        self.model.eval()
        y_pred, y_true, y_images, y_prob = [], [], [], [] 
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Calculate Predictions', disable=disable_pbar):
                images, labels = batch
                images, labels = images.to(self.dev), labels.to(self.dev)
                y_probs = F.softmax(self.model(images), dim = -1)
                
                if return_images:
                    y_images.append(images.cpu())
                y_prob.append(y_probs.cpu()) 
                y_true.append(labels.cpu())
                
        if return_images:
            y_images = torch.cat(y_images, dim = 0)
        
        y_prob = torch.cat(y_prob, dim = 0)
        y_true = torch.cat(y_true, dim = 0)
        y_pred = torch.argmax(y_prob, 1)        

        return (y_true, 
                y_pred, 
                y_images if return_images else None,
                y_prob if return_prob else None)
    
    def predict_one(self,imag:torch.Tensor,label_dict: dict, top_n: int = 1):
        """
        Returns true and predicted labels for prediction for one image
        Params:
        ---------
        model:           Pytorch Neuronal Net
        imag:            Image
        label_dict:      Dictionary of labels
        top_n:           Number of top predictions to return
        returns:
        ----------
        (y_true, y_pred, y_images, y_prob):
            y_true       True labels
            y_pred:      Predicted Labels
            y_prob:      Predicted Probability (empty if return_prob = False)
            y_images:    Images (empty if return_images = False)
        """
        
        self.model.to(self.dev)
        self.model.eval()
        
        image = imag.to(self.dev)
        y_prob = []
        with torch.no_grad():
            y_probs = F.softmax(self.model(image), dim = -1) 
            y_prob.append(y_probs.cpu()) 
        y_prob = torch.cat(y_prob, dim = 0)
        y_prob = y_prob[0]
        n_classes, n_probs = [], []
        example_prob, example_label = torch.sort(y_prob, dim=0)
        for i in range(top_n):
            label = example_label[-i-1]
            n_classes.append(list(label_dict.keys())[list(label_dict.values()).index(label)])
            n_probs.append(float(example_prob[-i-1]))
        zipped = list(zip(n_classes, n_probs))
        df = pd.DataFrame(zipped, columns=['Prediction', 'Probability'])
        df.index += 1 
        result = df.to_dict('index')
        return result

    
    def calculate_metrics(self, dataloader_train: 'torch.Dataloader', 
                          dataloader_test: 'torch.Dataloader', metric_funcs: list, 
                          disable_pbar: bool = False,
                          **metric_kwargs):
        """
        Calculates Metrics given functions

        Params:
        --------
        dataloader_train:          torch.Dataloader
                                   Batch Dataloader of Pytorch for Trainingset
        dataloader_test:           torch.Dataloader
                                   Batch Dataloader of Pytorch for Testset
        disable_pbar:              If true disables pbar
        metric_funcs:              list of functions (Preferably sklearn.metrics)

        return:
        ---------
        (metric1, metric2, ...): tuple
        """
        self.model.eval()
        with torch.no_grad():
            y_pred_train, y_true_train = [], []
            for batch in tqdm(dataloader_train, desc='Predictions Train-Set', disable=disable_pbar):
                images, labels = batch
                images, labels = images.to(self.dev), labels.to(self.dev)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                y_pred_train = np.append(y_pred_train, predicted.cpu().numpy())
                y_true_train = np.append(y_true_train, labels.cpu().numpy())

        # On testset
        with torch.no_grad():
            y_pred, y_true= [], []
            for batch in tqdm(dataloader_test, desc='Predictions Test-Set', disable=disable_pbar):
                images, labels = batch
                images, labels = images.to(self.dev), labels.to(self.dev)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                y_pred = np.append(y_pred, predicted.cpu().numpy())
                y_true = np.append(y_true, labels.cpu().numpy())

        metrics = {'train': {}, 'test': {}}
        for func in metric_funcs:
            train_score = func(y_true=y_true_train, y_pred=y_pred_train, **metric_kwargs)
            metrics['train'][func.__name__] = train_score
            test_score = func(y_true=y_true, y_pred=y_pred, **metric_kwargs)
            metrics['test'][func.__name__] = test_score
            if self.writer != None:
                self.writer.log({'train' + func.__name__: train_score})
                self.writer.log({'test' + func.__name__: test_score})

        return metrics

    
    
        
    