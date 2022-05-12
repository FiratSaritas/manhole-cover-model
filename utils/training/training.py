import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import KFold
import numpy as np
from tqdm.notebook import tqdm


def print_total_params(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(50 * '=')
    print(f'Trainable Parameters: {pytorch_total_params}')
    print(50 * '=')


def train_network(model, criterion, optimizer, n_epochs, dataloader_train, 
                  dataloader_val=None, verbose=True):
    """
    Trains a neural Network with given inputs and parameters.
    
    params:
    ---------
    model:                Neural Network class Pytorch     
    criterion:            Cost-Function used for the network optimizatio
    optimizer:            Optmizer for the network
    n_epochs:             Defines how many times the whole dateset should be fed through the network
    dataloader_train:     Dataloader with the batched dataset
    dataloader_val:       Dataloader test with the batched dataset used for test loop. If None -> No eval loop
    verbose               Prints Report after each Epoch
        
    returns:
    ----------
    tuple(model, losses):
        (resnetx, {'train': [3425, 12, 324], ...})
        model:
            trained model
        losses:
            Losses over each iteration  
    """
    y_pred, y_true = torch.Tensor(), torch.Tensor()
    train_losses, eval_losses = [], []
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(dev)
    criterion.to(dev)
    
    model.train()
    overall_length = len(dataloader_train)
    with tqdm(total=n_epochs*overall_length) as pbar:
        for epoch in range(n_epochs):  # loop over the dataset multiple times
            running_loss, val_loss = 0., 0.
            for i, data in enumerate(dataloader_train):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(dev), labels.to(dev)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                                    
                # calc and print stats
                train_losses.append(loss.item())
                running_loss += loss.item()                
                pbar.set_description('Epoch: {}/{} // Running Loss: {} '.format(epoch+1, n_epochs, 
                                                                                np.round(running_loss, 3)))
                pbar.update(1)
                
            if dataloader_val:
                length_dataloader_val = len(dataloader_val)
                val_loss = 0.
                for i, data in enumerate(dataloader_val):
                    pbar.set_description(f'Epoch: {epoch+1}/{n_epochs} // Eval-Loop: {i+1}/{length_dataloader_val}')
                    model.eval()
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data
                    inputs, labels = inputs.to(dev), labels.to(dev)
                    with torch.no_grad():
                        outputs = model(inputs)
                        eval_loss = criterion(outputs, labels)
                        val_loss += eval_loss.item()
                        eval_losses.append(eval_loss.item())
                    model.train()         
            if verbose:
                print('Epoch {}/{}: [Train-Loss = {}] || [Validation-Loss = {}]'.format(epoch+1, n_epochs, 
                                                                                     np.round(running_loss, 3),     
                                                                                     np.round(val_loss, 3)))                   
    return model, dict(train=train_losses, test=eval_losses)



def network_predict(model, dataloader):
    """
    Returns true and predicted labels for prediction

    params:
    ---------
    model:           Pytorch Neuronal Net
    dataloader:      batched Testset

    returns:
    ----------
    (y_true, y_pred): tuple
    y_true:
        True labels
    y_pred:
        Predicted Labels
    """
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(dev)
    model.eval()
    with torch.no_grad():
        y_pred = []
        y_true = []
        for batch in tqdm(dataloader, desc='Calculate Acc. on Test Data'):
            images, labels = batch
            images, labels = images.to(dev), labels.to(dev)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_pred = np.append(y_pred, predicted.cpu().numpy())
            y_true = np.append(y_true, labels.cpu().numpy())

    return (y_true, y_pred)


def calculate_metrics(model, dl_train, dl_test):
    """
    Calculates Train Accuracy and Test Accuracy

    params:
    --------
    model: torch.model
        Trained Pytorch Model
    dl_train: torch.Dataloader
        Batch Dataloader of Pytorch for Trainingset
    dl_test: torch.Dataloader
        Batch Dataloader of Pytorch for Testset

    return:
    ---------
    (train_acc, test_acc): tuple
        Training and Test Accuracy
    """
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.eval()
    with torch.no_grad():
        y_pred_train = []
        y_true_train = []
        for batch in tqdm(dl_train, desc='Calculate Acc. on Train Data'):
            images, labels = batch
            images, labels = images.to(dev), labels.to(dev)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_pred_train = np.append(y_pred_train, predicted.cpu().numpy())
            y_true_train = np.append(y_true_train, labels.cpu().numpy())

    # On testset
    with torch.no_grad():
        y_pred = []
        y_true = []
        for batch in tqdm(dl_test, desc='Calculate Acc. on Test Data'):
            images, labels = batch
            images, labels = images.to(dev), labels.to(dev)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_pred = np.append(y_pred, predicted.cpu().numpy())
            y_true = np.append(y_true, labels.cpu().numpy())

    train_acc = accuracy_score(y_true=y_true_train, y_pred=y_pred_train)
    test_acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    train_prec = precision_score(y_true=y_true_train, y_pred=y_pred_train, average='macro')
    test_prec = precision_score(y_true=y_true, y_pred=y_pred, average='macro')

    return dict(accuracy=[train_acc, test_acc], precision=[train_prec, test_prec])


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          train_data: 'ImageFolder'):
    label_translator = dict(
        zip([*train_data.class_to_idx.values()], [*train_data.class_to_idx.keys()])
    )
    label_translator
    plt.subplots(figsize=(14, 8))
    p = sns.heatmap(confusion_matrix(y_true=y_true, y_pred=y_pred), cmap='Blues',
                    xticklabels=label_translator.values(), yticklabels=label_translator.values())

    p.set_title('Confusion Matrix', loc='left')
    p.set_xlabel('Predicted')
    p.set_ylabel('True')

    plt.show()


def train_network_kfold(model, train_data: 'ImageFolder-Obj', test_data: 'ImageFolder-Obj',
                        criterion, optimizer, n_epochs, k_folds=10, batch_sizes=150, epsilon=None):
    """
    Wrapper Function for K-Fold Cross-Validation.
    """
    def reset_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()

    data = torch.utils.data.ConcatDataset([train_data, test_data])
    kfold = KFold(n_splits=k_folds, shuffle=True)

    metrics = {}
    for fold, (train_idx, test_idx) in enumerate(kfold.split(data)):
        metrics[fold] = {}
        print('{} Fold {} {}'.format(20 * '=', fold, 20 * '='))
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)

        trainloader = torch.utils.data.DataLoader(data, batch_size=batch_sizes, pin_memory=True,
                                                  sampler=train_subsampler, num_workers=20)
        testloader = torch.utils.data.DataLoader(data, batch_size=batch_sizes, pin_memory=True,
                                                 sampler=test_subsampler, num_workers=20)
        # Resetting weights for training of next fold
        model.apply(reset_weights)

        # Model Training
        model, losses, _ = train_network(model=model, criterion=criterion, optimizer=optimizer, n_epochs=n_epochs,
                                         dataloader_train=trainloader)
        metrics[fold]['losses'] = losses
        # Calculation of metrics
        print('Calculating Metrics ...')
        metrics[fold]['metrics'] = calculate_metrics(model=model, dl_train=trainloader, dl_test=testloader)

    return metrics