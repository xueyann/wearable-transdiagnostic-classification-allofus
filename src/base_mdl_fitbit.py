import pandas as pd
import numpy as np
import torch
import yaml
import os
import torch.optim as optim
import torch.nn as nn
import itertools

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
    roc_curve
)

from lib.transformer import SleepDataset, TransformerClassifier, ClassifierHead, \
    get_train_val_test_dataloader, compute_mean_and_std_batch, \
    compute_mean_and_std_static_batch, normalise_features, get_pred, \
    assess_clf_perf


def init_mdl(mdl_conf, input_dim, subcohort_cols):
    
    d_model = mdl_conf['d_model']
    n_layers = mdl_conf['n_layers']
    n_heads = mdl_conf['n_heads']
    d_ff = mdl_conf['d_ff']
    max_len = mdl_conf['max_len']
    dropout_p = mdl_conf['dropout_p']

    # Transformer model and classification head with only sleep features
    model = TransformerClassifier(input_dim = input_dim, 
                                            d_model = d_model, 
                                            n_layers = n_layers, 
                                            n_heads = n_heads, 
                                            d_ff = d_ff, 
                                            max_len = max_len,
                                            dropout_p = dropout_p)
    clf = ClassifierHead(d_model = d_model, out_dim = len(subcohort_cols))
    return model, clf


def mdl_train_loop(model, clf,
                mdl_conf, subcohort_cols, 
                train_dataloader, val_dataloader, test_dataloader,
                mean_features, std_features,
                mean_static, std_static,
                results_folder,
                with_static = False):


    ### Setting: time series only
    ts_only_config = mdl_conf['ts_only_config']
    max_epochs = ts_only_config['max_epoch']
    lr = ts_only_config['initial_lr']
    weight_decay = ts_only_config['weight_decay']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training using {device}")

    ######## Optimizer and learning rate scheduler ########
    optimizer = optim.AdamW(itertools.chain(model.parameters(),
                                            clf.parameters()), 
                            lr = lr, 
                            weight_decay = weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        **ts_only_config['cosine_anneal_lr_scheduler']
    )

    # Move model to device
    model = model.to(device)
    clf = clf.to(device)
    # define loss function
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5]).to(device))

    # Record the epoch losses for plotting
    epoch_losses = []
    epoch_val_losses = []
    epoch_val_aucs = []
    model_state_dicts = []
    clf_state_dicts = []

    # Put model back into training mode
    model.train()
    # Training loop
    for epoch in range(max_epochs):
    # for epoch in [0]:
        epoch_loss = 0
        for data, data_static, label in iter(train_dataloader):
            # Cast the data to float32 and normalise
            data, data_static = normalise_features(data, data_static, mean_features, std_features, mean_static, std_static)
            data = data.to(device)
            data_static = data_static.to(device)
            label = label.to(device)

            # Forward pass with transformer
            out = model(data)
            # # Fuse with static data
            # out = torch.concat((out, data_static), dim = 1)
            # Forward pass with classification head
            out = clf(out)
            # compute loss
            loss = loss_fn(out, label)
            
            # Backward and update
            optimizer.zero_grad()
            loss.backward()
            # # 1. Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.05)
            optimizer.step()
            # record epoch loss
            epoch_loss += loss.item()

        scheduler.step()
        epoch_loss /= len(train_dataloader)

        ######### Compute the validation loss #########
        model.eval()  # set to evaluation mode
        clf.eval()
        val_loss = 0.0
        val_probas = [] # also record the predicted probabilities
        val_true = []
        labels = []

        with torch.no_grad():  # disables gradient computation (saves memory & time)
            for data, data_static, label in val_dataloader:
                # Cast the data to float32 to ensure consistency with model parameters
                data, data_static = normalise_features(data, data_static, 
                                                       mean_features, std_features, 
                                                       mean_static, std_static)
                data = data.to(device)
                data_static = data_static.to(device)
                label = label.to(device)
                # Forward passes
                out = model(data)
                # out = torch.concat((out, data_static), dim = 1)
                out = clf(out)
                # compute loss
                loss = loss_fn(out, label)
                val_loss += loss.item()
                # Record the output probabilities
                val_probas.append(F.sigmoid(out).cpu())
                labels.append(label)
        val_loss /= len(val_dataloader)
        val_probas = torch.cat(val_probas, dim=0).to('cpu')
        val_true = torch.cat(labels).to('cpu')

        model.train() # put the model back to train mode
        clf.train()

        roc = roc_auc_score(val_true, val_probas)
            
        print(f"Epoch [{epoch+1}/{max_epochs}], Loss: {epoch_loss:.6f}, Validation loss: {val_loss:.6f}, AUC: {roc:.4f}, LR: {optimizer.param_groups[0]['lr']:.3e}")
        epoch_losses.append(epoch_loss)
        epoch_val_losses.append(val_loss)
        epoch_val_aucs.append(roc_auc_score(val_true, val_probas))

        # Save state dict once every epoch
        model_state_dicts.append(model.state_dict())
        clf_state_dicts.append(clf.state_dict())

        # Save the checkpoint
        checkpoint = {
                'epoch': epoch,
                'model_state_dict': model_state_dicts,
                'clf_state_dict': clf_state_dicts,
                'epoch_val_auc': epoch_val_aucs,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_losses,
                'val_loss': epoch_val_losses
            }
        
    torch.save(checkpoint, f'{results_folder}/checkpoint.pth')

    ###### plot training performance and evaluate model on test set #######
    # TODO put auc on secondary axis
    plt.plot(epoch_losses, label = 'Train loss')
    plt.plot(epoch_val_losses, label = 'Validation loss')
    plt.plot(epoch_val_aucs, label = 'Validation AUC Scores')
    plt.savefig(f'{results_folder}/train_and_val_losses.png', dpi = 300)

    test_pred, test_proba, test_labels = get_pred(model, clf, test_dataloader,
                                            mean_features,
                                            std_features,
                                            mean_static,
                                            std_static,
                                            device,
                                            with_static = False)

    for i in range(len(subcohort_cols)):
        subcohort = subcohort_cols[i]
        print(subcohort)
        #
        eval_folder = f'{results_folder}/{subcohort}'
        if not os.path.exists(eval_folder):
            os.makedirs(eval_folder)
        assess_clf_perf(test_pred[:,i], 
                        test_proba[:,i], 
                        test_labels[:,i], 
                        eval_folder,
                        labels = ['Negative (0)', 'Positive (1)'])


    return epoch_losses, epoch_val_losses, epoch_val_aucs



def main():
    config = yaml.safe_load(open('conf/base_mdl_fitbit.yaml'))
    ## read cohort and feature dataframes ##
    cohort = pd.read_csv(config['cohort'])
    subcohort_cols = [x for x in cohort.columns if x.startswith('subcohort_')]
    time_series_feature_df = pd.read_csv(config['time_series_feature_df'])
    time_series_sumstats = pd.read_csv(config['time_series_sumstats'])
    time_series_features = config['time_series_features']
    static_features = config['static_features']

    mdl_conf = config['transformer_mdl_setting']

    # get time series cohort for transformer model
    cohort_ts = cohort.merge(time_series_feature_df)


    dataset = SleepDataset(cohort_ts, 
                            features = time_series_features, 
                            date_col = 'date', 
                            label_col = subcohort_cols)
    # Split into dataloader
    g = torch.Generator().manual_seed(42)
    train_dataloader, val_dataloader, test_dataloader = get_train_val_test_dataloader(dataset, 0.15, 0.15, 128, g)
    # Compute mean and standard deviation across train set for normalisation later 
    mean_features, std_features = compute_mean_and_std_batch(train_dataloader)
    mean_static, std_static = compute_mean_and_std_static_batch(train_dataloader)


    # create results folder
    results_folder = mdl_conf['results_folder']
    results_folder_ts_only = f'{results_folder}/ts_only/'
    results_folder_with_static = f'{results_folder}/with_static/'
    # checkpoint_path_ts_only = f'{results_folder_ts_only}/checkpoint.pth'
    # checkpoint_path_with_static = f'{results_folder_with_static}/checkpoint.pth'

    # make dir
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    if not os.path.exists(results_folder_ts_only):
        os.makedirs(results_folder_ts_only)
    if not os.path.exists(results_folder_with_static):
        os.makedirs(results_folder_with_static)


    ##### time series feature only model #####
    input_dim = next(iter(dataset))[0].shape[1]
    model, clf = init_mdl(mdl_conf, input_dim, subcohort_cols)
    model, clf = mdl_train_loop(model, clf,
                                        mdl_conf, subcohort_cols, 
                                        train_dataloader, val_dataloader, test_dataloader,
                                        mean_features, std_features,
                                        mean_static, std_static,
                                        results_folder)
    # reinitialise classifier
    clf = ClassifierHead(d_model = mdl_conf['d_model']+len(static_features), out_dim = len(subcohort_cols))
    model, clf = mdl_train_loop(model, clf,
                                        mdl_conf, subcohort_cols, 
                                        train_dataloader, val_dataloader, test_dataloader,
                                        mean_features, std_features,
                                        mean_static, std_static,
                                        results_folder,
                                        with_static = True) # TODO implement with stastic True
    

    




    
