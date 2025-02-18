"""
Utility functions.
Author: patrick.mccarthy@dtc.ox.ac.uk
"""
import time
import torch
from torchvision import datasets, transforms
from itertools import product

def make_grid(param_dict):  
    param_names = param_dict.keys()
    combinations = product(*param_dict.values()) # creates list of all possible combinations of items in input list
    ds=[dict(zip(param_names, param_val)) for param_val in combinations] # convert to list of dicts
    return ds

def create_data_loaders(dataset, norm, batch_size, save_path, train_test_split=0.8):

    # choose normalisation method
    if norm == "normalise":
        transform = transforms.Compose([
            transforms.ToTensor(), # scale values to lie in range [0, 1]
        ])
    elif norm == "standardise":
                transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.), (1.,))  # standarise values to have zero mean and unit SD
        ])
                
    # load dataset and apply normalisation
    metadata = {}
    try:
        trainset = getattr(datasets, dataset)(root=save_path, train=True, download=True, transform=transform)
        testset = getattr(datasets, dataset)(root=save_path, train=False, download=True, transform=transform)
        metadata["classes"] = trainset.classes
    except Exception as e:
        print(f"Retrieving dataset failed with exception: {e}")

    # if dataset == "MNIST":
    #     trainset = datasets.MNIST(root=save_path,
    #                               train=True,
    #                               download=True,
    #                               transform=transform)
    #     testset = datasets.MNIST(root=save_path,
    #                               train=False,
    #                               download=True,
    #                               transform=transform)
    #     metadata["classes"] = trainset.classes
    # elif dataset == "FashionMNIST":
    #     trainset = datasets.FashionMNIST(root=save_path,
    #                                      train=True,
    #                                      download=True,
    #                                      transform=transform)
    #     testset = datasets.FashionMNIST(root=save_path,
    #                                     train=False,
    #                                     download=True,
    #                                     transform=transform)
        
    #     metadata["classes"] = trainset.classes

    # create loaders
    trainset_loader = torch.utils.data.DataLoader(dataset = trainset,
                                batch_size = batch_size,
                                shuffle = True)
    testset_loader = torch.utils.data.DataLoader(dataset = testset,
                                batch_size = batch_size,
                                shuffle = True)

    return trainset_loader, testset_loader, metadata

def train(model,
          trainset_loader,
          valset_loader,
          optimizer,
          loss_fn,
          ohe_targets,
          num_classes,
          num_epochs,
          device,
          loss_track_step):
    """Train model and regularly evaluate on validation set."""
    start_time = time.time()

    print("Training...")
    train_losses_epochs = []
    val_losses_epochs = []
    state_dicts = []
    for epoch in range(num_epochs):
        print(f"Beginning epoch {epoch+1}/{num_epochs}")
        train_losses, state_dict = train_one_epoch(
           model,
           trainset_loader,
           optimizer,
           loss_fn,
           ohe_targets,
           num_classes,
           device,
           loss_track_step,
           get_state_dict=True,
        )
        val_losses = evaluate(
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
        state_dicts.append(state_dict)
        print(f"Epoch {epoch+1}/{num_epochs} done")
    train_time = time.time() - start_time
    print(f"Finished training in {train_time:.2f} seconds.")

    return train_losses_epochs, val_losses_epochs, state_dicts, train_time 


def train_one_epoch(model, data_loader, optimizer, loss_fn, ohe_targets, num_classes, device, loss_track_step, get_state_dict=False):
    """Train model for one epoch."""
    size = len(data_loader.dataset)
    model.train()
    losses = []
    for batch, (X, y) in enumerate(data_loader):
        # move data to device where model is being trained
        X = X.to(device)
        y = y.to(device)
        
        # one-hot encode targets if specified
        if ohe_targets:
            y_one_hot = torch.nn.functional.one_hot(y, num_classes=num_classes).float()
        
        # autograd fwd pass
        y_est = model(X)

        # compute loss
        loss = loss_fn(y_est, y_one_hot)

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

    if get_state_dict:
        state_dict = model.state_dict()
    else:
        state_dict = None # return null value so number of return arguments always the same

    return losses, state_dict

def evaluate(model, data_loader, optimizer, loss_fn, ohe_targets, num_classes, device, loss_track_step):
    """Validate model for one epoch."""
    size = len(data_loader.dataset)
    model.train()
    losses = []
    with torch.no_grad():  # skip autograd tracking overhead

        for batch, (X, y) in enumerate(data_loader):
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

            # print loss every 500th batch
            if batch % loss_track_step == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(
                    f"validation batch {batch+1}, loss: {loss:.3f}, {current}/{size} datapoints"
                )
                losses.append(loss)

    return losses

groups={"group1": [1,2,3,4],
        "group2": [5,6,7,8]}


def train_dynamic_lr(model,
          trainset_loader,
          valset_loader,
          optimizer,
          loss_fn,
          ohe_targets,
          num_classes,
          num_epochs,
          device,
          loss_track_step,
          groups,
          learning_rates):
    """Train model and regularly evaluate on validation set."""
    start_time = time.time()

    print("Training...")
    train_losses_epochs = []
    val_losses_epochs = []
    for epoch in range(num_epochs):
        print(f"Beginning epoch {epoch+1}/{num_epochs}")
        train_losses = train_one_epoch_dynamic_lr(
           model,
           trainset_loader,
           optimizer,
           loss_fn,
           ohe_targets,
           num_classes,
           device,
           loss_track_step,
           groups,
           learning_rates
        )
        val_losses = evaluate(
           model,
           valset_loader,
           optimizer,
           loss_fn,
           ohe_targets,
           num_classes,
           device,
           loss_track_step
        )
        train_losses_epochs.append(train_losses)
        val_losses_epochs.append(val_losses)
        print(f"Epoch {epoch+1}/{num_epochs} done")
    train_time = time.time() - start_time
    print(f"Finished training in {train_time:.2f} seconds.")

    return train_losses_epochs, val_losses_epochs, train_time 


def train_one_epoch_dynamic_lr(model, data_loader, optimizer, loss_fn, ohe_targets, num_classes, device, loss_track_step, groups, learning_rates):
    """Train model for one epoch."""
    size = len(data_loader.dataset)
    model.train()
    losses = []
    for batch, (X, y) in enumerate(data_loader):
        # move data to device where model is being trained
        X = X.to(device)
        y = y.to(device)
        
        # one-hot encode targets if specified
        if ohe_targets:
            y_one_hot = torch.nn.functional.one_hot(y, num_classes=num_classes).float()
        
        # autograd fwd pass
        y_est = model(X)

        # compute loss
        loss = loss_fn(y_est, y_one_hot)

        # set learning rate based on class ID
        group = "none"
        for group_name, group_vals in groups.items():
            if y in group_vals:
                group = group_name
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rates[group])

        # autograd bwd pass
        loss.backward()  # compute gradients
        optimizer.step()  # update params
        optimizer.zero_grad()  # ensure not tracking gradients for next iteration

        # print loss every Nth batch
        if batch % loss_track_step == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(
                f"training batch {batch+1}, loss: {loss:.3f}, {current}/{size} datapoints"
            )
            losses.append(loss)

    return losses

class ModulatedOptimiser(torch.optim.SGD):
    """
    Custom optimiser with learning rates modulated by input, enabling plasticity gating to be modelled.
    """
    def __init__(self, params, lr=0.01):
        super().__init__(params, lr=lr)

    def step(self, mod_factors, closure=None):

        # scale gradients by modulation factors one-by-one
        for group in self.param_groups:
            for j, param in enumerate(group['params']):
                if param.grad is not None:
                    param.grad *= mod_factors[j] 
        
        # Perform regular optimizer step
        super().step(closure)


def train_one_epoch_plastic_mod(model, data_loader, optimizer, loss_fn, ohe_targets, num_classes, device, loss_track_step):
    """Train model for one epoch with plasticity gating."""
    size = len(data_loader.dataset)
    model.train()
    losses = []
    for batch, (X, y) in enumerate(data_loader):
        # move data to device where model is being trained
        X = X.to(device)
        y = y.to(device)
        
        # one-hot encode targets if specified
        if ohe_targets:
            y_one_hot = torch.nn.functional.one_hot(y, num_classes=num_classes).float()
        
        # autograd fwd pass
        y_est, thal = model(X)

        # compute loss
        loss = loss_fn(y_est, y_one_hot)

        # autograd bwd pass
        loss.backward()  # compute gradients
        optimizer.step(mod_factors=thal)  # update params
        optimizer.zero_grad()  # ensure not tracking gradients for next iteration

        # print loss every Nth batch
        if batch % loss_track_step == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(
                f"training batch {batch+1}, loss: {loss:.3f}, {current}/{size} datapoints"
            )
            losses.append(loss)

    return losses