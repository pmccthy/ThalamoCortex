"""
Train a grid of CTCNet models with "driver-type" thalamocortical projections.
Author: patrick.mccarthy@dtc.ox.ac.uk
"""

import os
import sys
import logging
import argparse
import traceback
from datetime import date
from copy import copy, deepcopy
from pathlib import Path

import numpy as np
import torch
import pickle
import wandb

from thalamocortex.models import CTCNet
from thalamocortex.utils import make_grid, create_data_loaders, train, evaluate

# Create a logger
logger = logging.getLogger("MyLogger")
logger.setLevel(logging.DEBUG)  # Set logging level

# Custom class to redirect stdout to logger
class LoggerWriter:
    def __init__(self, level):
        self.level = level  # Logging level (INFO, ERROR, etc.)

    def write(self, message):
        if message.strip():  # Avoid logging empty lines
            self.level(message.strip())

    def flush(self):
        pass  # Needed for compatibility with sys.stdout

# Save paths
data_save_path = "/users/costa/abz308/ThalamoCortex/thalamocortex/data"
results_save_path = "/users/costa/abz308/ThalamoCortex/thalamocortex/results"

# hyperparameter grid for driver-type model
hyperparam_grid = {
    # data hyperparams
    "norm" : ["normalise"],
    "dataset" : ["MNIST"],
    "save_path" : ["/Users/patmccarthy/Documents/thalamocortex/data"],
    "batch_size" : [32],
    # model hyperparams
    "input_size" : [28 * 28],
    "output_size" : [10],
    "ctx_layer_size" : [32],
    "thal_layer_size" : [16],
    "thalamocortical_type" : [None],
    "thal_reciprocal" : [False], 
    "thal_to_readout" : [False], 
    "thal_per_layer" : [False],
    # training hyperparams
    "lr" : [1e-6],
    "loss" : [torch.nn.CrossEntropyLoss()],
    "epochs": [10],
    "ohe_targets": [True],
    "track_loss_step": [50]
}

if __name__ == "__main__":
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        prog="CTCNet model grid training script.",
        description="Trains grid of CTCNet models on MNIST with specified hyperparams.",
    )
    parser.add_argument("-n", "--name")  # option that takes a value
    args = parser.parse_args()

    # Set save directory (and make if does not exist)
    save_path_this_run = Path(results_save_path, f"{args.name}")
    if not os.path.exists(save_path_this_run):
        os.mkdir(save_path_this_run)

    # Log file handler
    log_file_path = Path(save_path_this_run, "training.log")
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)

    # Console file handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    # Create a logging formatter and set it for both handlers
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Redirect stdout and stderr to logger
    sys.stdout = LoggerWriter(logger.info)
    sys.stderr = LoggerWriter(logger.error)

    # Make parameter grid
    model_param_grid = make_grid(hyperparam_grid)

    # Set backend
    logger.info("Setting backend.")
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    logger.info(f"Using {device} device.")

    # number of parameter combinations
    num_comb = len(model_param_grid)
    for hp_comb_idx, hyperparams in enumerate(model_param_grid):
        
        # Start a new wandb run to track this script.
        run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="pmccthy-university-of-oxford",
            # Set the wandb project where this run will be logged.
            project="thalamocortex",
            # Track hyperparameters and run metadata.
            config=deepcopy(hyperparams),
            name=f"{str(os.path.basename(__file__)).split('.py')[0]}_hp{hp_comb_idx}"
        ) 
        logger.info(f"Running hyperparameter combination {hp_comb_idx+1} of {num_comb}")

        # create readable tag for saving
        if hp_comb_idx < 10:
            comb_id = f"0{hp_comb_idx}"
        else:
            comb_id = copy(hp_comb_idx)
        tag = f"{hp_comb_idx}_CTCNet"
        if hyperparams["thalamocortical_type"] is None:
            tag += "_TC_none"
        else:
            tag += f"_TC_{hyperparams['thalamocortical_type']}"
        if hyperparams["thal_reciprocal"]:
            tag += "_reciprocal"
        if hyperparams["thal_to_readout"]:
            tag += "_readout"
        if hyperparams["thal_per_layer"]:
            tag += "_per_layer"
        
        # print model tag
        logger.info(f"{tag}")

        # create path for saving this model
        save_path_this_model = Path(save_path_this_run, tag)
        if not os.path.exists(save_path_this_model):
            os.mkdir(save_path_this_model)
                
        try:

            # create data loaders
            logger.info("Loading data...")
            trainset_loader, testset_loader, metadata = create_data_loaders(dataset=hyperparams["dataset"],
                                                                            norm=hyperparams["norm"],
                                                                            save_path=hyperparams["save_path"],
                                                                            batch_size=hyperparams["batch_size"])
            logger.info("Done loading.")

            # create model
            logger.info("Building model and optimiser...")
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
            logger.info("Done.")

            # train model
            train_losses, val_losses, train_topk_accs, val_topk_accs, state_dicts, train_time = train(model=model,
                                            trainset_loader=trainset_loader,
                                            valset_loader=testset_loader,
                                            optimizer=optimizer,
                                            loss_fn=loss_fn,
                                            ohe_targets=hyperparams["ohe_targets"],
                                            num_classes=len(metadata["classes"]),
                                            num_epochs=hyperparams["epochs"],
                                            device=device,
                                            loss_track_step=hyperparams["track_loss_step"],
                                            wandb_run=run)
            logger.info("Model trained in {train_time:.2f} s")

            # evaluate model
            logger.info("Evaluating model...")
            losses, topk_accs = evaluate(model=model,
                            data_loader=testset_loader,
                            optimizer=optimizer,
                            loss_fn=loss_fn,
                            ohe_targets=hyperparams["ohe_targets"],
                            num_classes=len(metadata["classes"]),
                            device=device,
                            loss_track_step=hyperparams["track_loss_step"])
            logger.info("Done evaluating.")

            # take average of final validation loss
            final_val_loss_avg = np.mean(losses)
            logger.info(f"Average final validation loss: {final_val_loss_avg:.3f}")

            # Save model
            logger.info("Saving...")
            # model
            torch.save(model.state_dict(), Path(f"{save_path_this_model}", "model.pth"))
            # hyperparams
            with open(Path(f"{save_path_this_model}", "hyperparams.pkl"), "wb") as handle:
                pickle.dump(hyperparams, handle)
            # learning progress
            training_stats = {"train_losses": train_losses,
                              "val_losses": val_losses,
                              "train_topk_accs": train_topk_accs,
                              "val_topk_accs": val_topk_accs,
                              "final_val_losses": losses,
                              "final_val_topk_accs": topk_accs,
                              "state_dicts": state_dicts,
                              "train_time": train_time}
            with open(Path(f"{save_path_this_model}", "learning.pkl"), "wb") as handle:
                pickle.dump(training_stats, handle)
            logger.info("Done saving.")

            logger.info(f"Successfully completed hyperparameter combination {hp_comb_idx+1} of {num_comb}")
        
            # finish wandb run and upload remaining data
            run.finish()

        except Exception as e:
            logger.info(f"Failed hyperparameter combination {hp_comb_idx+1} of {num_comb} with exception: {e}")
