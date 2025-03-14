from pathlib import Path
import pickle
import glob
import os
import copy
import traceback

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colormaps as cm
from scipy.special import softmax
from scipy.linalg import svd
from sklearn import svm
from sklearn.decomposition import KernelPCA, PCA
from scipy.stats import entropy

from thalamocortex.models import CTCNet
from thalamocortex.utils import create_data_loaders, activation_hook, get_neuron_weights, plot_receptive_field

plt.rcParams['legend.fontsize'] = 8  
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.facecolor'] = 'w'
plt.rcParams['axes.facecolor'] = 'w'  
plt.rcParams['savefig.dpi'] = 300 
plt.rcParams['savefig.bbox'] = 'tight'  
plt.rcParams['savefig.pad_inches'] = 0.1 
plt.rcParams['axes.titlesize'] = 10  
plt.rcParams['figure.titlesize'] = 10  

save_path = "/Users/patmccarthy/Documents/phd/rotation1/results_11_03_25/leftrightmnist"

# Load results
results_paths = {
    "no feedback": "/Users/patmccarthy/Documents/thalamocortex/results/11_03_25_feedforward_leftrightmnist/2_CTCNet_TC_none",
    "driver": "/Users/patmccarthy/Documents/thalamocortex/results/11_03_25_driver_leftrightmnist/2_CTCNet_TC_add_reciprocal_readout",
    "modulator": "/Users/patmccarthy/Documents/thalamocortex/results/11_03_25_mod1_leftrightmnist/2_CTCNet_TC_multi_pre_activation_reciprocal",
}

if __name__ == "__main__":

    # load models, learning stats, results 
    results = {}
    for model_name, path in results_paths.items():
        
        # NOTE: note loading trained models because can instantiate from final weights

        # hyperparameters
        with open(Path(f"{path}", "hyperparams.pkl"), "rb") as handle:
            hp = pickle.load(handle)

        # learning progress
        with open(Path(f"{path}", "learning.pkl"), "rb") as handle:
            learning = pickle.load(handle)

        # store results and params in dict
        results[model_name] = {"val_losses": learning["val_losses"],
                            "train_losses": learning["train_losses"],
                            "val_topk_accs": learning["val_topk_accs"],
                            "train_topk_accs": learning["train_topk_accs"],
                            "train_time": learning["train_time"],
                            "state_dicts": learning["state_dicts"],
                            "hyperparams": hp}
        
        # get number of epochs to train for
        n_epochs = len(learning["train_topk_accs"])

        if model_name in ["no feedback", "driver", "modulator"]:
            results[model_name]["val_losses"].pop(0)
            results[model_name]["val_topk_accs"].pop(0)
            results[model_name]["state_dicts"].pop(0)

        else:
            epoch_range = np.arange(0, n_epochs)

        # get top-1 accuracies in more convenient form for plotting
        train_top1_accs = []
        val_top1_accs = []
        
        # store training info
        for epoch in np.arange(n_epochs):
            train_top1_accs.append(learning["train_topk_accs"][epoch][1])
            val_top1_accs.append(learning["val_topk_accs"][epoch][1])
        results[model_name]["train_top1_accs"] = np.array(train_top1_accs)
        results[model_name]["val_top1_accs"] = np.array(val_top1_accs)
        
    ### **Learning progress**
    model_plot_list = ["no feedback", "driver", "modulator"]
    # loss through time
    # n_epochs = len(results[model_plot_list[0]]["val_losses"])
    colours = ["r", "m", "b", "o"]
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    models_plotted = []
    models_plotted_idx = 0
    for _, (model_name, model_results) in enumerate(results.items()):
        if model_name in model_plot_list:
            models_plotted.append(model_name)
            models_plotted_idx += 1

            # ax.plot(np.arange(n_epochs+1), np.median(np.array(model_results["val_losses"]), axis=-1), ls="--", linewidth=1, label=f"{model_name} val.", c=colours[models_plotted_idx-1])
            # ax.plot(1+np.arange(n_epochs), np.median(np.array(model_results["train_losses"]), axis=-1), ls="-", linewidth=1, label=f"{model_name} train", c=colours[models_plotted_idx-1])

            ax.plot(np.arange(n_epochs), np.median(np.array(model_results["val_losses"]), axis=-1), ls="--", linewidth=1, label=f"{model_name} val.", c=colours[models_plotted_idx-1])
            ax.plot(np.arange(n_epochs), np.median(np.array(model_results["train_losses"]), axis=-1), ls="-", linewidth=1, label=f"{model_name} train", c=colours[models_plotted_idx-1])

    # ax.set_xticks(range(1, len(models_plotted)+1), models_plotted)
    ax.set_ylabel("cross entropy")
    ax.set_xlabel("epoch")
    ax.set_xlim(0, n_epochs)
    ax.set_ylim(0, 2.5)
    ax.legend(loc="upper right")
    fig.savefig(Path(save_path, "loss_curve.png"))
    # accuracy through time
    fig, ax = plt.subplots(1, 1, figsize=(10, 3), layout="constrained")
    ax.axhline(10, ls="--", c="k", label="chance")
    models_plotted = []
    models_plotted_idx = 0
    for _, (model_name, model_results) in enumerate(results.items()):
        if model_name in model_plot_list:
            models_plotted.append(model_name)
            models_plotted_idx += 1

            ax.plot(np.arange(n_epochs), np.array(model_results["val_top1_accs"]) * 100,ls="--", label=f"{model_name} val.", linewidth=1, c=colours[models_plotted_idx-1])
            ax.plot(np.arange(n_epochs), np.array(model_results["train_top1_accs"]) * 100, ls="-", label=f"{model_name} train", linewidth=1, c=colours[models_plotted_idx-1])

    # ax.set_xticks(range(1, len(models_plotted)+1), models_plotted)
    ax.set_ylabel("top-1 accuracy (%)")
    ax.set_xlabel("epoch")
    ax.set_ylim(0, 100)
    ax.set_xlim(0, n_epochs)
    ax.legend(loc="center right")
    fig.savefig(Path(save_path, "accuracy_curve.png"))
    # accuracy before and after convergence
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.axhline(10, ls="--", c="k", label="chance")
    for models_plotted_idx, (model_name, model_results) in enumerate(results.items()):
        # print(f"{models_plotted_idx}")
        if model_name in model_plot_list:
            ax.plot([0, 1], [model_results["train_top1_accs"][0] * 100, model_results["train_top1_accs"][-1] * 100], c=colours[models_plotted_idx], marker="o", markersize=10, linewidth=4, label=model_name)
    ax.set_ylabel("test top-1 accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.25, 1.25)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["before", "after"])
    ax.legend(loc="upper left")
    fig.savefig(Path(save_path, "accuracy_prepostlearning.png"))
    ### **Trained model analysis**
    models_selected = ["no feedback", "driver", "modulator"]
    # epoch of trained model weights to use 
    epoch_trained = 800
    # Test set inference
    # create loaders
    trainset_loader, testset_loader, metadata = create_data_loaders(dataset=results[models_selected[0]]["hyperparams"]["dataset"],
                                                                    norm=results[models_selected[0]]["hyperparams"]["norm"],
                                                                    batch_size=32,
                                                                    save_path="/Users/patmccarthy/Documents/ThalamoCortex/data")

    # load full test set
    X_all = []
    y_all = []
    for X, y in iter(testset_loader):
        # X_all.append(X.detach().numpy()[:, 0, :, :])
        # y_all.append(y.detach().numpy()[:])
        X_all.append(X[:, :, :])
        y_all.append(y[:])
    if results[models_selected[0]]["hyperparams"]["dataset"] in ["BinaryMNIST", "LeftRightMNIST"]:
        # Concatenate along the first axis (num_samples)
        X_all_arr = np.concatenate(X_all, axis=0)  # Shape: (num_samples, 1, 28, 28)
        y_all_reshaped = np.concatenate(y_all, axis=0)  # Shape: (num_samples,)

        # Reshape X to [samples, features]
        X_all_reshaped = X_all_arr.reshape(X_all_arr.shape[0], -1)  # Shape: (num_samples, 28*28)
    else:
        X_all_tensor = torch.cat(X_all, dim=0)  # Shape: [num_samples, 1, 28, 28]
        y_all_tensor = torch.cat(y_all, dim=0)  # Shape: [num_samples]

        # Convert to NumPy
        X_all_arr = X_all_tensor.numpy()  # Shape: (num_samples, 1, 28, 28)
        y_all_reshaped = y_all_tensor.numpy()  # Shape: (num_samples,)

        # Reshape X to [samples, features]
        X_all_reshaped = X_all_arr.reshape(X_all_arr.shape[0], -1)  # Shape: (num_samples, 28*28)

    sides = metadata["sides"]
    sides_encoded = np.zeros(len(sides))
    sides_encoded[np.where(np.array(sides) == "left")[0]] = 1
    sides_encoded = sides_encoded[:X_all_reshaped.shape[0]]

    # # Decoding class ID
    # # inference on full test set using models trained to various epochs
    # epochs_range = np.arange(0, 800, 25)
    # epochs_range = np.append(epochs_range, [799])
    # activations = {}
    # for model_selected in models_selected:
    #     activations[model_selected] = {}

    #     # instantiate model
    #     model = CTCNet(input_size=results[model_selected]["hyperparams"]["input_size"],
    #                 output_size=results[model_selected]["hyperparams"]["output_size"],
    #                 ctx_layer_size=results[model_selected]["hyperparams"]["ctx_layer_size"],
    #                 thal_layer_size=results[model_selected]["hyperparams"]["thal_layer_size"],
    #                 thalamocortical_type=results[model_selected]["hyperparams"]["thalamocortical_type"],
    #                 thal_reciprocal=results[model_selected]["hyperparams"]["thal_reciprocal"],
    #                 thal_to_readout=results[model_selected]["hyperparams"]["thal_to_readout"], 
    #                 thal_per_layer=results[model_selected]["hyperparams"]["thal_per_layer"])
        
    #     for epoch in epochs_range:
    #         activations[model_selected][epoch] = {}

    #         # get model trained to specified epoch
    #         weights = results[model_selected]["state_dicts"][epoch]

    #         # set model weights
    #         model.load_state_dict(weights)

    #         # Register hooks for specific layers
    #         hook_handles = []
    #         activations_this_epoch = {}
    #         for name, layer in model.named_modules():
    #             handle = layer.register_forward_hook(lambda module, input, output: activation_hook(module, input, output, activations_this_epoch))
    #             hook_handles.append(handle)
            
    #         # inference (on full dataset)
    #         with torch.no_grad():
                
    #             y_est_logits = model(torch.Tensor(X_all_reshaped))
    #             y_est_prob = softmax(y_est_logits.detach().numpy())
    #             y_est = np.argmax(y_est_prob, axis=1)

    #             # Remove hooks after use
    #             for handle in hook_handles:
    #                 handle.remove()
            
    #         activations[model_selected][epoch] = copy.deepcopy(activations_this_epoch)
    # # define readable names for connections of interest
    # # NOTE: always double check these before usimng
    # readable_names = {"ctx1": list(activations["ff_MNIST"][0].keys())[2],
    #                   "ctx2": list(activations["ff_MNIST"][0].keys())[5],
    #                   "ctx_readout": list(activations["ff_MNIST"][0].keys())[7]
    #                 #   "thal": list(activations["ff_MNIST"][0].keys())[10], # TODO: figure out why thal layer not showing up in activations dict
    # }
    # readable_layer_idxs = {"ctx1": 2,
    #                     "ctx2": 5,
    #                     "ctx_readout": 7,
    #                     "thal": 10, # TODO: figure out why thal layer not showing up in activations dict
    # }
    # # activations decoding analysis 
    # layers_selected = ["ctx1", "ctx2", "thal"]
    # train_test_split = 0.8
    # accuracies = {}
    # for layer_selected in layers_selected:
    #     accuracies[layer_selected] = {}
    #     for model_selected in models_selected:
    #         print(f"Decoding for {layer_selected}, {model_selected}")
    #         accuracies[layer_selected][model_selected] = {}
    #         for epoch in epochs_range:
                
    #             try:

    #                 # select layer activations to decode from
    #                 features = activations[model_selected][epoch][list(activations[model_selected][epoch].keys())[readable_layer_idxs[layer_selected]]].detach().numpy()

    #                 # split into train and test set
    #                 test_cutoff = int(len(features) * train_test_split)
    #                 X_train = features[:test_cutoff, :]
    #                 X_test = features[test_cutoff:, :]
    #                 y_train = y_all_reshaped[:test_cutoff]
    #                 y_test = y_all_reshaped[test_cutoff:]

    #                 # TODO: perform cross-validation
    #                 # TODO: try replacing with linear classifier
                    
    #                 # train SVM classifier
    #                 clf = svm.SVC(kernel="linear")
    #                 clf.fit(X_train, y_train)

    #                 # test SVM classifier
    #                 y_pred = clf.predict(X_test)
                    
    #                 # compute classification accuracy
    #                 correct = 0
    #                 for samp_idx in range(y_pred.shape[0]):
    #                     if y_pred[samp_idx] == y_test[samp_idx]:
    #                         correct += 1
    #                 accuracy = correct / y_pred.shape[0]
    #                 print(f"epoch: {epoch}, accuracy: {accuracy * 100:.2f}%")

    #                 accuracies[layer_selected][model_selected][epoch] = accuracy
    #             except:
    #                 accuracies[layer_selected][model_selected][epoch] = np.nan

    # # save decoding accuracies
    # with open(Path(save_path, "svm_decoding_id_acc.pkl"), "wb") as handle:
    #     pickle.dump(accuracies, handle)
    # # load decoding accuracies if don't want to re-run
    # with open(Path(save_path, "svm_decoding_id_acc.pkl"), "rb") as handle:
    #     accuracies = pickle.load(handle)

    # for layer in accuracies.keys():
    #     fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    #     for model_idx, model_selected in enumerate(accuracies[layer].keys()):
        #     ax.plot(accuracies[layer][model_selected].keys(), np.array(list(accuracies[layer][model_selected].values())) * 100, c=colours[model_idx], label=model_selected)
        # ax.axhline(10, ls="--", c="k", label="chance")
        # ax.set_title(f"SVM decoding MNIST from {layer} activations")
        # ax.set_ylabel("classification accuracy (%)")
        # ax.set_xlabel("epoch")
        # ax.set_ylim(0, 100)
        # ax.set_xlim(list(accuracies[layer][model_selected].keys())[0], list(accuracies[layer][model_selected].keys())[-1])
        # ax.legend(loc="center right")
        # fig.savefig(Path(save_path, f"svm_decoding_{layer}.png"))

    # Decoding cue side
    # inference on full test set using models trained to various epochs
    epochs_range = np.arange(0, 800, 25)
    epochs_range = np.append(epochs_range, [799])
    activations = {}
    print(f"{models_selected=}")
    for model_selected in models_selected:
        activations[model_selected] = {}
        print(f"{model_selected=}")
        # instantiate model
        model = CTCNet(input_size=results[model_selected]["hyperparams"]["input_size"],
                    output_size=results[model_selected]["hyperparams"]["output_size"],
                    ctx_layer_size=results[model_selected]["hyperparams"]["ctx_layer_size"],
                    thal_layer_size=results[model_selected]["hyperparams"]["thal_layer_size"],
                    thalamocortical_type=results[model_selected]["hyperparams"]["thalamocortical_type"],
                    thal_reciprocal=results[model_selected]["hyperparams"]["thal_reciprocal"],
                    thal_to_readout=results[model_selected]["hyperparams"]["thal_to_readout"], 
                    thal_per_layer=results[model_selected]["hyperparams"]["thal_per_layer"])
        
        for epoch in epochs_range:
            activations[model_selected][epoch] = {}

            # get model trained to specified epoch
            weights = results[model_selected]["state_dicts"][epoch]

            # set model weights
            model.load_state_dict(weights)

            # Register hooks for specific layers
            hook_handles = []
            activations_this_epoch = {}
            for name, layer in model.named_modules():
                handle = layer.register_forward_hook(lambda module, input, output: activation_hook(module, input, output, activations_this_epoch))
                hook_handles.append(handle)
            
            # inference (on full dataset)
            with torch.no_grad():
                
                y_est_logits = model(torch.Tensor(X_all_reshaped))
                y_est_prob = softmax(y_est_logits.detach().numpy())
                y_est = np.argmax(y_est_prob, axis=1)

                # Remove hooks after use
                for handle in hook_handles:
                    handle.remove()
            
            activations[model_selected][epoch] = copy.deepcopy(activations_this_epoch)


    # define readable names for connections of interest
    # NOTE: always double check these before usimng
    # readable_names = {"ctx1": list(activations["ff_MNIST"][0].keys())[2],
    #                   "ctx2": list(activations["ff_MNIST"][0].keys())[5],
    #                   "ctx_readout": list(activations["ff_MNIST"][0].keys())[7]
    #                 #   "thal": list(activations["ff_MNIST"][0].keys())[10], # TODO: figure out why thal layer not showing up in activations dict
    # }
    readable_layer_idxs = {"ctx1": 2,
                        "ctx2": 5,
                        "ctx_readout": 7,
                        "thal": 10, # TODO: figure out why thal layer not showing up in activations dict
    }
    # activations decoding analysis 
    layers_selected = ["ctx1", "ctx2", "thal"]
    print(f"{activations.keys()=}")
    train_test_split = 0.8
    accuracies = {}
    for layer_selected in layers_selected:
        accuracies[layer_selected] = {}
        for model_selected in models_selected:
            print(f"Decoding for {layer_selected}, {model_selected}")
            accuracies[layer_selected][model_selected] = {}
            for epoch in epochs_range:
                
                try:

                    # select layer activations to decode from
                    features = activations[model_selected][epoch][list(activations[model_selected][epoch].keys())[readable_layer_idxs[layer_selected]]].detach().numpy()

                    # split into train and test set
                    test_cutoff = int(len(features) * train_test_split)
                    X_train = features[:test_cutoff, :]
                    X_test = features[test_cutoff:, :]

                    # y_train = y_all_reshaped[:test_cutoff]
                    # y_test = y_all_reshaped[test_cutoff:]
                    y_train = sides_encoded[:test_cutoff]
                    y_test = sides_encoded[test_cutoff:]
                    
                    # TODO: perform cross-validation
                    # TODO: try replacing with linear classifier
                    
                    # train SVM classifier
                    clf = svm.SVC(kernel="linear")
                    clf.fit(X_train, y_train)

                    # test SVM classifier
                    y_pred = clf.predict(X_test)
                    
                    # compute classification accuracy
                    correct = 0
                    for samp_idx in range(y_pred.shape[0]):
                        if y_pred[samp_idx] == y_test[samp_idx]:
                            correct += 1
                    accuracy = correct / y_pred.shape[0]
                    print(f"epoch: {epoch}, accuracy: {accuracy * 100:.2f}%")

                    accuracies[layer_selected][model_selected][epoch] = accuracy
                except: # Exception as e:
                    # print(f"Failed ad decoding {layer_selected}, {model_selected} with exception: {e}")
                    # traceback.print_exc(e)
                    # accuracies[layer_selected][model_selected][epoch] = np.nan
                    pass
    
    # save decoding accuracies
    with open(Path(save_path, "svm_decoding_side_acc2.pkl"), "wb") as handle:
        pickle.dump(accuracies, handle)
    # # load decoding accuracies if don't want to re-run
    # with open(Path(save_path, "svm_decoding_side_acc.pkl"), "rb") as handle:
    #     accuracies = pickle.load(handle)