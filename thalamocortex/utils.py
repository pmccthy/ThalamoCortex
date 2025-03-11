"""
Utility functions.
Author: patrick.mccarthy@dtc.ox.ac.uk
"""
import copy
import time
import pickle
from pathlib import Path
import numpy as np
import torch
from torchvision import datasets, transforms
from itertools import product

def make_grid(param_dict):  
    param_names = param_dict.keys()
    combinations = product(*param_dict.values()) # creates list of all possible combinations of items in input list
    ds=[dict(zip(param_names, param_val)) for param_val in combinations] # convert to list of dicts
    return ds

class CustomDataLoader(tuple):
    """"
    Mock of torch DataLoader class to enable custom dataset loading.
    (additional attribute batch size)
    """
    def __new__(cls, values, batch_size):
        return super().__new__(cls, values)

    def __init__(self, values, batch_size):
        self.values = values
        self.batch_size = batch_size

def create_data_loaders_custom_datasets(path, batch_size, normalise="normalise"):
    """
    Create data loader for LeftRightMNIST which is analagous to torch loaders. 
    """
    metadata = {"classes": np.arange(10)} # TODO: set based on dataset instead of hardcode
    loaders = {"train": None, "test": None}
    for dataset_type in loaders.keys():

        # load trainset
        with open(Path(path, f"{dataset_type}.pkl"), "rb") as handle:
            dataset = pickle.load(handle)

        # choose number of batches
        num_batches = dataset["X"].shape[0] // batch_size
        
        if dataset == "LeftRightMNIST":
            metadata["sides"] = dataset["sides"]
            
        # create loaders
        data_loader = []
        for batch_idx in range(num_batches):
            if normalise == "normalise":
                X_batch = normalise_data(dataset["X"][batch_idx * batch_size: (batch_idx+1) * batch_size, :, :])
                y_batch = dataset["y"][batch_idx * batch_size: (batch_idx+1) * batch_size]
            elif normalise == "standardise":
                X_batch = standardise_data(dataset["X"][batch_idx * batch_size: (batch_idx+1) * batch_size, :, :])
                y_batch = dataset["y"][batch_idx * batch_size: (batch_idx+1) * batch_size]
            elif normalise is None:
                X_batch = dataset["X"][batch_idx * batch_size: (batch_idx+1) * batch_size, :, :]
                y_batch = dataset["y"][batch_idx * batch_size: (batch_idx+1) * batch_size]
            else:
                raise Exception(f"Unrecognised normalisation method '{normalise}'. Choose 'normalise', 'standardise', or None.")
            data_loader.append((X_batch, y_batch))
        data_loader = CustomDataLoader(data_loader, batch_size=batch_size)
        loaders[dataset_type] = data_loader

    return loaders["train"], loaders["test"], metadata

def normalise_data(x):
    return (x - np.min(x))/(np.max(x) - np.min(x))

def standardise_data(x):
    return x - np.mean(x)/np.var(x)

def create_data_loaders(dataset, norm, batch_size, save_path):

    metadata = {}
    if dataset in ["LeftRightMNIST", "BinaryMNIST"]:
        path = Path(save_path, dataset)
        trainset_loader, testset_loader, metadata = create_data_loaders_custom_datasets(path, batch_size, normalise=norm)
    else:
        if dataset == "CIFAR10":
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor()  
            ])

        # choose normalisation method
        if norm == "normalise":
            if dataset == "CIFAR10":
                transform = transforms.Compose([
                    transforms.Grayscale(num_output_channels=1), # convert to greyscale if CIFAR10 chosen
                    transforms.ToTensor(), # scale values to lie in range [0, 1]
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(), # scale values to lie in range [0, 1]
                ])
        elif norm == "standardise":
            if dataset == "CIFAR10":
                transform = transforms.Compose([
                    transforms.Grayscale(num_output_channels=1), # convert to greyscale if CIFAR10 chosen
                    transforms.ToTensor(), 
                    transforms.Normalize((0.), (1.,))  # standarise values to have zero mean and unit SD
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(), 
                    transforms.Normalize((0.), (1.,))  # standarise values to have zero mean and unit SD
                ])    
                    
        # load dataset and apply normalisation
        try:
            trainset = getattr(datasets, dataset)(root=save_path, train=True, download=True, transform=transform)
            testset = getattr(datasets, dataset)(root=save_path, train=False, download=True, transform=transform)
            metadata["classes"] = trainset.classes
        except Exception as e:
            print(f"Retrieving dataset failed with exception: {e}")

        # create loaders
        trainset_loader = torch.utils.data.DataLoader(dataset = trainset,
                                    batch_size = batch_size,
                                    shuffle = True)
        testset_loader = torch.utils.data.DataLoader(dataset = testset,
                                    batch_size = batch_size,
                                    shuffle = True)

    if dataset == "BinaryMNIST":    
        metadata["classes"] = np.arange(2)

    return trainset_loader, testset_loader, metadata

def add_cue_patch(img, cue_params, side):
    """
    Add cue patch to supplied image with specified params.
    """

    # get params and generate patch pixels
    img_dims = img.shape
    location = cue_params[side]["location"]
    cue_patch = np.tile(cue_params[side]["intensity"], reps=[cue_params["size"][0],cue_params["size"][1]])

    # generate cue pixel coordinaes
    if location == "tl":
        px_range = [[cue_params["margin"][0], cue_params["margin"][0]+cue_params["size"][0]],
                    [cue_params["margin"][1], cue_params["margin"][1]+cue_params["size"][1]]]
    elif location == "bl":
        px_range = [[img_dims[0] - cue_params["size"][0] - cue_params["margin"][0], img_dims[0] - cue_params["margin"][0]],
                    [cue_params["margin"][1], cue_params["margin"][1]+cue_params["size"][1]]]
    elif location == "tr":
        px_range = [[cue_params["margin"][0], cue_params["margin"][0]+cue_params["size"][0]],
                    [img_dims[1] - cue_params["size"][1] - cue_params["margin"][1], img_dims[1] - cue_params["margin"][1]]]
    elif location == "br":
        px_range = [[img_dims[0] - cue_params["size"][0] - cue_params["margin"][0], img_dims[0] - cue_params["margin"][0]],
                    [img_dims[1] - cue_params["size"][1] - cue_params["margin"][1], img_dims[1] - cue_params["margin"][1]]]
        
    # add cue patch to image
    img[px_range[0][0]:px_range[0][1], px_range[1][0]:px_range[1][1]] = cue_patch

    return img

def train(model,
          trainset_loader,
          valset_loader,
          optimizer,
          loss_fn,
          ohe_targets,
          num_classes,
          num_epochs,
          device,
          loss_track_step,
          get_state_dict=False,
          wandb_run=None,
          topk=(1,5)):
    """Train model and regularly evaluate on validation set."""
    start_time = time.time()

    print("Training...")
    train_losses_epochs = []
    val_losses_epochs = []
    train_topk_accs_epochs = []
    val_topk_accs_epochs = []
    state_dicts = []
    # perform initial evaluation and store results
    val_losses, val_topk_accs = evaluate(
           model,
           valset_loader,
           optimizer,
           loss_fn,
           ohe_targets,
           num_classes,
           device,
           loss_track_step,
        )
    state_dict = copy.deepcopy(model.state_dict())
    val_losses_epochs.append(val_losses)
    val_topk_accs_epochs.append(val_topk_accs)
    state_dicts.append(state_dict)
    for epoch in range(num_epochs):
        print(f"Beginning epoch {epoch+1}/{num_epochs}")
        train_losses, train_topk_accs, state_dict = train_one_epoch(
           model,
           trainset_loader,
           optimizer,
           loss_fn,
           ohe_targets,
           num_classes,
           device,
           loss_track_step,
           get_state_dict=True,
           topk=topk
        )
        val_losses, val_topk_accs = evaluate(
           model,
           valset_loader,
           optimizer,
           loss_fn,
           ohe_targets,
           num_classes,
           device,
           loss_track_step,
        )
        train_losses_epochs.append(train_losses)
        val_losses_epochs.append(val_losses)
        train_topk_accs_epochs.append(train_topk_accs)
        val_topk_accs_epochs.append(val_topk_accs)
        state_dicts.append(state_dict)
        # create string for printing performance summary
        topk_acc_str = ""
        for k, v in train_topk_accs.items():
            topk_acc_str += f"top-{k} acc: {val_topk_accs[k]:.3f} "
        print(f"Epoch {epoch+1}/{num_epochs} done.")
        print(
            f"Final validation performance:\nLoss: {np.mean(val_losses):.3f}, {topk_acc_str}"
            )
        # create dictionary of performance metrics for logging to wandb rubn
        wandb_log = {"train_loss": np.mean(train_losses),
                     "val_loss": np.mean(val_losses)}
        for k, v in train_topk_accs.items():
            wandb_log[f"top{k}_acc"]= val_topk_accs[k]
        # log to wandb
        if wandb_run is not None:
            wandb_run.log(wandb_log)
            
    train_time = time.time() - start_time
    print(f"Finished training in {train_time:.2f} seconds.")

    return train_losses_epochs, val_losses_epochs, train_topk_accs_epochs, val_topk_accs_epochs, state_dicts, train_time 


def train_one_epoch(model, data_loader, optimizer, loss_fn, ohe_targets, num_classes, device, loss_track_step, get_state_dict=False, topk=(1,5)):
    """Train model for one epoch."""
    size = len(data_loader) * data_loader.batch_size
    model.train()
    losses = []
    topk_correct = {k: 0 for k in topk}
    total_samples = 0
    for batch, (X, y) in enumerate(data_loader):
        
        # cast to torch array if necessary
        if type(X) is not torch.Tensor:
            X = torch.from_numpy(X).to(torch.float32)
            y = torch.from_numpy(y).to(torch.int64)

        # move data to device where model is being trained
        X = X.to(device)
        y = y.to(device)
        
        # one-hot encode targets if specified
        if ohe_targets:
            y_one_hot = torch.nn.functional.one_hot(y, num_classes=num_classes).float()
        
        # fwd pass
        y_est = model(X)

        # compute loss
        loss = loss_fn(y_est, y_one_hot)

        # compute top-k accuracy
        for k in topk:
            _, topk_pred = y_est.topk(k, dim=1)
            topk_correct[k] += sum([y[i] in topk_pred[i] for i in range(y.size(0))])
        total_samples += y.size(0)
        
        # autograd bwd pass
        loss.backward()  # compute gradients
        optimizer.step()  # update params
        optimizer.zero_grad()  # ensure not tracking gradients for next iteration

        # get loss
        loss, current = loss.item(), (batch + 1) * len(X)
        losses.append(loss)

        # print loss every Nth batch
        if batch % loss_track_step == 0:
            print(
                f"training batch {batch+1}, loss: {loss:.3f}, {current}/{size} datapoints"
            )
    
    topk_acc = {k: topk_correct[k] / total_samples for k in topk}

    if get_state_dict:
        state_dict = copy.deepcopy(model.state_dict())
    else:
        state_dict = None # return null value so number of return arguments always the same

    return losses, topk_acc, state_dict

def evaluate(model, data_loader, optimizer, loss_fn, ohe_targets, num_classes, device, loss_track_step, topk=(1,5)):
    """Validate model for one epoch."""
    size = len(data_loader) * data_loader.batch_size
    model.train()
    losses = []
    topk_correct = {k: 0 for k in topk}
    total_samples = 0
    with torch.no_grad():  # skip autograd tracking overhead

        for batch, (X, y) in enumerate(data_loader):

            # cast to torch array if necessary  
            if type(X) is not torch.Tensor:
                X = torch.from_numpy(X).to(torch.float32)
                y = torch.from_numpy(y).to(torch.int64)

            # move data to device where model is being trained
            X = X.to(device)
            y = y.to(device)

            # one-hot encode targets if specified
            if ohe_targets:
                y_one_hot = torch.nn.functional.one_hot(y, num_classes=num_classes).float()

            # autograd fwd pass            
            y_est = model(X) 

            # compute loss
            loss = loss_fn(y_est, y_one_hot)

            # compute top-k accuracy
            topks = []
            for k in topk:
                _, topk_pred = y_est.topk(1, dim=1)
                topk_correct[k] += sum([y[i] in topk_pred[i] for i in range(y.size(0))])
                topks.append(topk_pred)
            total_samples += y.size(0)
        
            # print loss every 500th batch
            if batch % loss_track_step == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(
                    f"validation batch {batch+1}, loss: {loss:.3f}, {current}/{size} datapoints"
                )
                losses.append(loss)

    topk_acc = {k: topk_correct[k] / total_samples for k in topk}

    return losses, topk_acc

def evaluate_thalreadout(model, data_loader, optimizer, loss_fn, ohe_targets, num_classes, device, loss_track_step, topk=(1,5)):
    """
    TODO: fill in function for evaluating model with thalamic readout
    """
    """Validate model for one epoch."""
    size = len(data_loader) * data_loader.batch_size
    model.train()
    losses = []
    topk_correct = {k: 0 for k in topk}
    total_samples = 0
    with torch.no_grad():  # skip autograd tracking overhead

        for batch, (X, y) in enumerate(data_loader):

            # cast to torch array if necessary  
            if type(X) is not torch.Tensor:
                X = torch.from_numpy(X).to(torch.float32)
                y = torch.from_numpy(y).to(torch.int64)

            # move data to device where model is being trained
            X = X.to(device)
            y = y.to(device)

            # one-hot encode targets if specified
            if ohe_targets:
                y_one_hot = torch.nn.functional.one_hot(y, num_classes=num_classes).float()

            # autograd fwd pass            
            _, y_est = model(X)

            # compute loss
            loss = loss_fn(y_est, y_one_hot)

            # compute top-k accuracy
            topks = []
            for k in topk:
                _, topk_pred = y_est.topk(1, dim=1)
                topk_correct[k] += sum([y[i] in topk_pred[i] for i in range(y.size(0))])
                topks.append(topk_pred)
            total_samples += y.size(0)

            # print loss every 500th batch
            if batch % loss_track_step == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(
                    f"validation batch {batch+1}, loss: {loss:.3f}, {current}/{size} datapoints"
                )
                losses.append(loss)
    
    topk_acc = {k: topk_correct[k] / total_samples for k in topk}

    return losses, topk_acc

def train_one_epoch_thalreadout(model,
                                data_loader,
                                optimizer,
                                loss_fn,
                                ohe_targets,
                                num_classes,
                                device,
                                loss_track_step,
                                get_state_dict=False,
                                topk=(1,5)):
    """
    Train thalamic readout model for one epoch.
    Fine-tuning using thalamic loss only.
    TODO: implement trainer for 
    """
    size = len(data_loader) * data_loader.batch_size
    model.train()
    losses = []
    topk_correct = {k: 0 for k in topk}
    total_samples = 0
    for batch, (X, y) in enumerate(data_loader):
        
        # cast to torch array if necessary
        if type(X) is not torch.Tensor:
            X = torch.from_numpy(X).to(torch.float32)
            y = torch.from_numpy(y).to(torch.int64)

        # move data to device where model is being trained
        X = X.to(device)
        y = y.to(device)
        
        # one-hot encode targets if specified
        if ohe_targets:
            y_one_hot = torch.nn.functional.one_hot(y, num_classes=num_classes).float()
        
        # fwd pass
        _, y_est = model(X)

        # compute loss
        loss = loss_fn(y_est, y_one_hot)

        # compute top-k accuracy
        topks = []
        for k in topk:
            _, topk_pred = y_est.topk(1, dim=1)
            topk_correct[k] += sum([y[i] in topk_pred[i] for i in range(y.size(0))])
            topks.append(topk_pred)
        total_samples += y.size(0)

        # autograd bwd pass
        loss.backward()  # compute gradients
        optimizer.step()  # update params
        optimizer.zero_grad()  # ensure not tracking gradients for next iteration

        # get loss
        loss, current = loss.item(), (batch + 1) * len(X)
        losses.append(loss)

        # print loss every Nth batch
        if batch % loss_track_step == 0:
            print(
                f"training batch {batch+1}, loss: {loss:.3f}, {current}/{size} datapoints"
            )

    topk_acc = {k: topk_correct[k] / total_samples for k in topk}

    if get_state_dict:
        state_dict = copy.deepcopy(model.state_dict())
    else:
        state_dict = None # return null value so number of return arguments always the same

    return losses, topk_acc, state_dict


def train_thalreadout(model,
                      trainset_loader,
                      valset_loader,
                      optimizer,
                      loss_fn,
                      ohe_targets,
                      num_classes,
                      num_epochs,
                      device,
                      loss_track_step,
                      get_state_dict=False,
                      wandb_run=None,
                      topk=(1,5)):
    """Train model and regularly evaluate on validation set."""
    start_time = time.time()

    print("Training...")
    train_losses_epochs = []
    val_losses_epochs = []
    train_topk_accs_epochs = []
    val_topk_accs_epochs = []
    state_dicts = []
    # perform initial evaluation and store results
    val_losses, val_topk_accs = evaluate(
           model,
           valset_loader,
           optimizer,
           loss_fn,
           ohe_targets,
           num_classes,
           device,
           loss_track_step,
        )
    state_dict = copy.deepcopy(model.state_dict())
    val_losses_epochs.append(val_losses)
    val_topk_accs_epochs.append(val_topk_accs)
    state_dicts.append(state_dict)
    for epoch in range(num_epochs):
        print(f"Beginning epoch {epoch+1}/{num_epochs}")
        train_losses, train_topk_accs, state_dict = train_one_epoch_thalreadout(
           model,
           trainset_loader,
           optimizer,
           loss_fn,
           ohe_targets,
           num_classes,
           device,
           loss_track_step,
           get_state_dict=True,
           topk=topk
        )
        val_losses, val_topk_accs = evaluate_thalreadout(
           model,
           valset_loader,
           optimizer,
           loss_fn,
           ohe_targets,
           num_classes,
           device,
           loss_track_step,
        )
        train_losses_epochs.append(train_losses)
        val_losses_epochs.append(val_losses)
        train_topk_accs_epochs.append(train_topk_accs)
        val_topk_accs_epochs.append(val_topk_accs)
        state_dicts.append(state_dict)
        # create string for printing performance summary
        topk_acc_str = ""
        for k, v in train_topk_accs.items():
            topk_acc_str += f"top-{k} acc: {val_topk_accs[k]:.3f} "
        print(f"Epoch {epoch+1}/{num_epochs} done.")
        print(
            f"Final validation performance:\nLoss: {np.mean(val_losses):.3f}, {topk_acc_str}"
            )
        # create dictionary of performance metrics for logging to wandb rubn
        wandb_log = {"train_loss": np.mean(train_losses),
                     "val_loss": np.mean(val_losses)}
        for k, v in train_topk_accs.items():
            wandb_log[f"top{k}_acc"]= val_topk_accs[k]
        # log to wandb
        if wandb_run is not None:
            wandb_run.log(wandb_log)
            
        print(f"Epoch {epoch+1}/{num_epochs} done")
    
    train_time = time.time() - start_time
    print(f"Finished training in {train_time:.2f} seconds.")

    return train_losses_epochs, val_losses_epochs, train_topk_accs_epochs, val_topk_accs_epochs, state_dicts, train_time 


def train_one_epoch_thalreadout_ctx_readout(model,
                                            data_loader,
                                            optimizer,
                                            loss_fn,
                                            ohe_targets,
                                            num_classes,
                                            device,
                                            loss_track_step,
                                            get_state_dict=False):
    """
    Train thalamic readout model for one epoch.
    Fine-tuning using thalamic loss only.
    TODO: implement trainer for 
    """
    size = len(data_loader) * data_loader.batch_size
    model.train()
    losses_ctx = []
    losses_thal = []
    losses = {"ctx": [], "thal": [], "combined": []}
    for batch, (X, y_thal, y_ctx) in enumerate(data_loader):
        
        # cast to torch array if necessary
        if type(X) is not torch.Tensor:
            X = torch.from_numpy(X).to(torch.float32)
            y_ctx = torch.from_numpy(y_ctx).to(torch.int64)
            y_thal = torch.from_numpy(y_thal).to(torch.int64)

        # move data to device where model is being trained
        X = X.to(device)
        y_ctx = y_ctx.to(device)
        y_thal = y_thal.to(device)

        # one-hot encode targets if specified
        if ohe_targets:
            y_ctx_one_hot = torch.nn.functional.one_hot(y_ctx, num_classes=num_classes).float()
            y_thal_one_hot = torch.nn.functional.one_hot(y_thal, num_classes=num_classes).float()

        # fwd pass
        y_ctx_est, y_thal_est = model(X)

        # compute loss
        loss_ctx = loss_fn(y_ctx_est, y_ctx_one_hot)
        loss_thal = loss_fn(y_thal_est, y_thal_one_hot)
        loss = loss_ctx + loss_thal

        # autograd bwd pass
        loss.backward()  # compute gradients
        optimizer.step()  # update params
        optimizer.zero_grad()  # ensure not tracking gradients for next iteration

        # get loss
        loss, current = loss.item(), (batch + 1) * len(X)
        loss_ctx, _ = loss_ctx.item(), (batch + 1) * len(X)
        loss_thal, _ = loss_thal.item(), (batch + 1) * len(X)

        losses["ctx"].append(loss_ctx)
        losses["thal"].append(loss_thal)
        losses["combined"].append(loss)

        # print loss every Nth batch
        if batch % loss_track_step == 0:
            print(
                f"training batch {batch+1}, loss: {loss:.3f}, {current}/{size} datapoints"
            )

    if get_state_dict:
        state_dict = copy.deepcopy(model.state_dict())
    else:
        state_dict = None # return null value so number of return arguments always the same

    return losses, state_dict

def activation_hook(module, input, output, activations):
    activations[module] = output

# Custom class to redirect stdout to logger
class LoggerWriter:
    def __init__(self, level):
        self.level = level  # Logging level (INFO, ERROR, etc.)

    def write(self, message):
        if message.strip():  # Avoid logging empty lines
            self.level(message.strip())

    def flush(self):
        pass  # Needed for compatibility with sys.stdout

def get_neuron_weights(weights, neuron_id, shape=[28, 56]):
    weights_neuron = weights[neuron_id, :].detach().numpy()
    weights_neuron_reshaped = np.reshape(weights_neuron, newshape=shape, order="C")
    return weights_neuron_reshaped

def plot_receptive_field(weights, ax, cmap, clims, title=None):
    ax.imshow(weights, cmap=cmap)
    ax.set_title(title)
    ax.set_xticks([], [])
    ax.set_yticks([], [])