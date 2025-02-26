"""
A script for fine-tuning a pretrained model with a thalamic readout.
Author: patrick.mccarthy@dtc.ox.ac.uk
"""

import os
import copy
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

from thalamocortex.models import CTCNetThalReadout
from thalamocortex.utils import make_grid, create_data_loaders, train_thalreadout, evaluate_thalreadout, LoggerWriter

# Create a logger
logger = logging.getLogger("MyLogger")
logger.setLevel(logging.DEBUG)  # Set logging level

# Save paths
data_save_path = "/Users/patmccarthy/Documents/thalamocortex/data"
results_save_path = "/Users/patmccarthy/Documents/thalamocortex/results"

# hyperparameter grid
hyperparam_grid = {
    # data hyperparams
    "norm" : ["normalise"],
    "dataset" : ["BinaryMNIST"],
    "save_path" : ["/Users/patmccarthy/Documents/thalamocortex/data"],
    "batch_size" : [32],
    # model hyperparams
    "thal_output_size": [2],
    "pretrained_epoch": [200],
    "pretrained_model_path": ["/Users/patmccarthy/Documents/thalamocortex/results/25_02_24_feedforward_mnist_gridsearch/0_CTCNet_TC_none/model.pth"],
    # training hyperparams
    "lr" : [5e-6],
    "loss" : [torch.nn.CrossEntropyLoss()],
    "epochs": [800],
    "ohe_targets": [True],
    "track_loss_step": [50]
}

if __name__  == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        prog="CTCNetThalReadout model grid training script (for finetuning CTCNet-trained model).",
        description="Trains grid of CTCNetThalReadout models on MNIST with specified hyperparams.",
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

    # set backend
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Make parameter grid
    model_param_grid = make_grid(hyperparam_grid)

    num_comb = len(model_param_grid)
    for hp_comb_idx, hyperparams in enumerate(model_param_grid):
        
        logger.info(f"Running hyperparameter combination {hp_comb_idx+1} of {num_comb}")

        # create readable tag for saving
        if hp_comb_idx < 10:
            comb_id = f"0{hp_comb_idx}"
        else:
            comb_id = copy(hp_comb_idx)
        tag = f"{hp_comb_idx}_CTCNet_finetuning"
        tag += f"_ThalReadout_{hyperparams['thal_output_size']}"

        
        # print model tag
        logger.info(f"{tag}")

        # create path for saving this model
        save_path_this_model = Path(save_path_this_run, tag)
        if not os.path.exists(save_path_this_model):
            os.mkdir(save_path_this_model)

        try:

            # pretrained model path
            pretrained_model_path = Path(hyperparams["pretrained_model_path"])

            # get training stats and hyperparameter paths
            learning_path = Path(hyperparams["pretrained_model_path"].split("/model.pth")[0], "learning.pkl")
            params_path = Path(hyperparams["pretrained_model_path"].split("/model.pth")[0], "hyperparams.pkl")

            # load training progress
            with open(learning_path, "rb") as handle:
                learning = pickle.load(handle)
                print(f"{learning.keys()=}")
            results = {"val_losses": learning["val_losses"],
                       "train_losses": learning["train_losses"],
                       "train_time": learning["train_time"],
                       "state_dicts": learning["state_dicts"]}

            # load hyperparams
            with open(params_path, "rb") as handle:
                hyperparams_pretrained = pickle.load(handle)

            # retrieve weights of pretrained model for specified epoch
            weights = results["state_dicts"][hyperparams["pretrained_epoch"]]

            # create hyperparams for new model
            hyperparams["pretrained_params"] = deepcopy(hyperparams_pretrained)

            # create model with thalamic readout
            model_thal = CTCNetThalReadout(input_size=hyperparams["pretrained_params"]["input_size"],
                                           ctx_output_size=hyperparams["pretrained_params"]["output_size"],
                                           thal_output_size=hyperparams["thal_output_size"],
                                           ctx_layer_size=hyperparams["pretrained_params"]["ctx_layer_size"],
                                           thal_layer_size=hyperparams["pretrained_params"]["thal_layer_size"],
                                           thalamocortical_type=hyperparams["pretrained_params"]["thalamocortical_type"],
                                           thal_reciprocal=hyperparams["pretrained_params"]["thal_reciprocal"],
                                           thal_to_readout=hyperparams["pretrained_params"]["thal_to_readout"], 
                                           thal_per_layer=hyperparams["pretrained_params"]["thal_per_layer"])
            
            # ger initial model weights
            model_thal_init_weights = model_thal.state_dict()

            # add these weights to backbone weights object
            thal_unique_params = []
            for param_name in model_thal_init_weights.keys():
                print(param_name)
                if param_name not in list(weights.keys()):
                    thal_unique_params.append(param_name)
                    weights[param_name] = model_thal_init_weights[param_name]

            # set model weights
            model_thal.load_state_dict(weights)
                    
            # load data
            trainset_loader, testset_loader, metadata = create_data_loaders(hyperparams["dataset"], "normalise", 32, "/Users/patmccarthy/Documents/thalamocortex/data")

            # define loss and optimiser
            loss_fn = deepcopy(hyperparams["loss"])
            optimizer = torch.optim.Adam(model_thal.parameters(),
                                        lr = hyperparams["lr"])
            
            # train model
            train_losses, val_losses, state_dicts, train_time = train_thalreadout(model=model_thal,
                                                                                            trainset_loader=trainset_loader,
                                                                                            valset_loader=testset_loader,
                                                                                            optimizer=optimizer,
                                                                                            loss_fn=loss_fn,
                                                                                            ohe_targets=hyperparams["ohe_targets"],
                                                                                            num_classes=2,
                                                                                            num_epochs=hyperparams["epochs"],
                                                                                            device=device,
                                                                                            loss_track_step=hyperparams["track_loss_step"],
                                                                                            get_state_dict=True)

            # evaluate model
            logger.info(f"Evaluating model...")
            losses = evaluate_thalreadout(model=model_thal,
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
            torch.save(model_thal.state_dict(), Path(f"{save_path_this_model}", "model.pth"))
            # hyperparams
            with open(Path(f"{save_path_this_model}", "hyperparams.pkl"), "wb") as handle:
                pickle.dump(hyperparams, handle)
            # learning progress
            training_stats = {"train_losses": train_losses,
                            "val_losses": val_losses,
                            "final_val_losses": losses,
                            "state_dicts": state_dicts,
                            "train_time": train_time}
            with open(Path(f"{save_path_this_model}", "learning.pkl"), "wb") as handle:
                pickle.dump(training_stats, handle)
            logger.info("Done saving.")

            logger.info(f"Successfully completed hyperparameter combination {hp_comb_idx+1} of {num_comb}")
        
        except Exception as e:
            logger.info(f"Failed hyperparameter combination {hp_comb_idx+1} of {num_comb} with exception: {e}")
