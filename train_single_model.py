"""
Train a single CTCNet model on FashionMNIST.
"""

import pickle
import os
from copy import deepcopy
from pathlib import Path
import torch
from models import CTCNet
from utils import create_data_loaders, train, evaluate

# Set backend
print("Setting backend.")
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using {device} device.")

data_save_path = "/Users/patmccarthy/Documents/thalamocortex/data"
results_save_path = "/Users/patmccarthy/Documents/thalamocortex/results"

hyperparams = {
    # data hyperparams
    "norm" : "normalise",
    "dataset" : "FashionMNIST",
    "save_path" : "/Users/patmccarthy/Documents/thalamocortex/data",
    "batch_size" : 32,
    # model hyperparams
    "input_size" : 28 * 28,
    "output_size" : 10,
    "ctx_layer_size" : 128,
    "thal_layer_size" : 64,
    "thalamocortical_type" : "add",
    "thal_reciprocal" : True, # True or False
    "thal_to_readout" : True, # True or False
    "thal_per_layer" : False, # if False, mixing from cortical layers
    # training hyperparams
    "lr" : 0.001,
    "loss" : torch.nn.CrossEntropyLoss(),
    "epochs": 100,
    "ohe_targets": True,
    "track_loss_step": 50
}

if __name__ == "__main__":
    # create data loaders
    print("Loading data...")
    trainset_loader, testset_loader, metadata = create_data_loaders(dataset=hyperparams["dataset"],
                                                                            norm=hyperparams["norm"],
                                                                            save_path=hyperparams["save_path"],
                                                                            batch_size=hyperparams["batch_size"])
    print("Done loading.")

    # create model
    print("Building model and optimiser...")
    model = CTCNet(input_size=hyperparams["input_size"],
                    output_size=hyperparams["output_size"],
                    ctx_layer_size=hyperparams["ctx_layer_size"],
                    thal_layer_size=hyperparams["thal_layer_size"],
                    thalamocortical_type=hyperparams["thalamocortical_type"],
                    thal_reciprocal=hyperparams["thal_reciprocal"],
                    thal_to_readout=hyperparams["thal_to_readout"], 
                    thal_per_layer=hyperparams["thal_per_layer"])
    model.summary()

    # define loss and optimiser
    loss_fn = deepcopy(hyperparams["loss"])
    optimizer = torch.optim.Adam(model.parameters(),
                                lr = hyperparams["lr"])
    print("Done.")

    # train model
    train_losses, val_losses, train_time = train(model=model,
                                    trainset_loader=trainset_loader,
                                    valset_loader=testset_loader,
                                    optimizer=optimizer,
                                    loss_fn=loss_fn,
                                    ohe_targets=hyperparams["ohe_targets"],
                                    num_classes=len(metadata["classes"]),
                                    num_epochs=hyperparams["epochs"],
                                    device=device,
                                    loss_track_step=hyperparams["track_loss_step"])
    
    # evaluate model
    print("Evaluating model...")
    losses = evaluate(model=model,
                    data_loader=testset_loader,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    ohe_targets=hyperparams["ohe_targets"],
                    num_classes=len(metadata["classes"]),
                    device=device,
                    loss_track_step=hyperparams["track_loss_step"])
    print("Done evaluating.")

    # Save model
    save_path_this_model = Path(results_save_path, "model0_05_01_24")
    if not os.path.exists(save_path_this_model):
        os.mkdir(save_path_this_model)
    print("Saving...")
    # model
    torch.save(model.state_dict(), Path(f"{save_path_this_model}", "model.pth"))
    # hyperparams
    with open(Path(f"{save_path_this_model}", "hyperparams.pkl"), "wb") as handle:
        pickle.dump(hyperparams, handle)
    # learning progress
    training_stats = {"train_losses": train_losses,
                    "val_losses": val_losses,
                    "final_val_losses": losses,
                    "train_time": train_time}
    with open(Path(f"{save_path_this_model}", "learning.pkl"), "wb") as handle:
        pickle.dump(training_stats, handle)
    print("Done saving.")