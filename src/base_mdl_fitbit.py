import sys
sys.path.append('./')
import pandas as pd
import numpy as np
import torch
import yaml
import os
import torch.optim as optim
import torch.nn as nn
import itertools
import torch.nn.functional as F
from functools import reduce
import multiprocessing
import re
from functools import partial

import matplotlib.pyplot as plt

import shap
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
    roc_curve,
    brier_score_loss,
    precision_recall_curve
)
from sklearn.calibration import calibration_curve
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from lib.transformer import SleepDataset, TransformerClassifier, ClassifierHead, \
    get_train_val_test_dataloader, compute_mean_and_std_batch, \
    compute_mean_and_std_static_batch, normalise_features, get_pred, \
    assess_clf_perf
from lib.helpers import format_p
from src.mdl_sleep import assess_clf_perf as assess_clf_perf_xgb
from imblearn.over_sampling import RandomOverSampler

# Modelling
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

from lib.model import bootstrap_fmt, paired_bootstrap_fmt, delong_fmt
import pickle


def scaled_brier_score(y_true, y_prob):
    brier_score = brier_score_loss(y_true, y_prob)
    prevalence = y_true.sum()/len(y_true)
    return 1 - brier_score/prevalence

def get_net_benefit(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    N = len(y_true)
    
    return (TP / N) - (FP / N) * (threshold / (1 - threshold))

def plot_decision_curve(y_true, y_prob_dict, ax, title = "Decision Curve Analysis"):
    thresholds = np.linspace(0.01, 0.7, 69)
    
    # Reference values
    event_rate = np.mean(y_true)
    nb_treat_all = [
        event_rate - (1 - event_rate) * (t / (1 - t))
        for t in thresholds
    ]
    nb_treat_none = np.zeros_like(thresholds)

    for mdl_name, y_prob in y_prob_dict.items():
        nb_model = [get_net_benefit(y_true, y_prob, t) for t in thresholds]
        ax.plot(thresholds, nb_model, label = mdl_name)
    # plot reference values
    ax.plot(thresholds, nb_treat_all, linestyle='--', label="Treat all")
    ax.plot(thresholds, nb_treat_none, linestyle=':', label="Treat none")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title(title)
    ax.legend()

    return ax

def plot_calibration_curve(y_true, y_prob_dict, ax, title = "Calibration Curve"):

    for mdl_name, y_prob in y_prob_dict.items():
        prob_true, prob_pred = calibration_curve(
            y_true,
            y_prob,
            n_bins=10,
            strategy="uniform"  # or "uniform"
            )
        ax.plot(prob_pred, prob_true, marker='o', label=mdl_name)

    # plot perfect calibration
    ax.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration')
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed event rate")
    ax.legend()
    ax.set_title(title)

    return ax 

def get_perf_diff(y_true, y_prob_base, y_prob, 
                  threshold_base = 0.5, 
                  threshold_fb = 0.5):

    
    def accuracy_score_modified(y_true, y_prob, threshold=0.5): 
        return accuracy_score(y_true, (y_prob >= threshold).astype(int))
    def precision_score_modified(y_true, y_prob, threshold=0.5): 
        return precision_score(y_true, (y_prob >= threshold).astype(int))
    def recall_score_modified(y_true, y_prob, threshold=0.5): 
        return recall_score(y_true, (y_prob >= threshold).astype(int))
    def f1_score_modified(y_true, y_prob, threshold=0.5): 
        return f1_score(y_true, (y_prob >= threshold).astype(int))


    metric_dict = {
        'Accuracy': accuracy_score_modified,
        'Precision': precision_score_modified,
        'Recall': recall_score_modified,
        'F1 Score': f1_score_modified,
        'ROC AUC Score': roc_auc_score,
        'Brier Score': brier_score_loss,
        'Brier Skill Score (BSS)': scaled_brier_score
    }

    # Get classification metrics
    perf_dict_base = {metric: bootstrap_fmt(y_true, y_prob_base, partial(func, threshold = threshold_base)) \
                            if metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score'] else \
                            bootstrap_fmt(y_true, y_prob_base, func) for metric, func in metric_dict.items()}
    perf_dict = {metric: bootstrap_fmt(y_true, y_prob, partial(func, threshold = threshold_fb)) \
                            if metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score'] else \
                            bootstrap_fmt(y_true, y_prob, func) for metric, func in metric_dict.items()}
    perf_dict_diff_p = {metric: paired_bootstrap_fmt(y_true, y_prob_base, y_prob, func) \
                        if metric != 'ROC AUC Score' else delong_fmt(y_true, y_prob_base, y_prob) \
                        for metric, func in metric_dict.items()}
    
    return perf_dict_base, perf_dict, perf_dict_diff_p

def comp_perf_logreg_xgb(y_true, y_prob_logreg, y_prob_xgb, mdl_name, 
                         threshold_logreg, threshold_xgb):
    _, _, perf_dict_diff_p = get_perf_diff(y_true, y_prob_logreg, y_prob_xgb,
                                           threshold_logreg, threshold_xgb)

    perf = pd.concat([pd.Series({k:v[0] for k,v in perf_dict_diff_p.items()}),
                      pd.Series({k:v[1] for k,v in perf_dict_diff_p.items()})],
                      axis = 1,
                      keys = [(mdl_name, 'Difference (95% CI)'),
                              (mdl_name, 'p-value')])

    return perf
    

def get_clf_metrics(y_true, y_prob_base, y_prob, mdl_name, ax1, threshold_base, threshold_fb): 
    """
    Docstring for get_clf_metrics
    
    :param y_true: Description
    :param y_prob_base: predicted probability from base model
    :param y_prob: predicted probability from full model
    :param mdl_name: Description
    :param ax1: axes where probability density will be plotted
    :param ax2: axes where calibration curve will be plotted
    :param threshold: Description


    """

    perf_dict_base, perf_dict, perf_dict_diff_p = get_perf_diff(y_true, y_prob_base, y_prob, threshold_base, threshold_fb)

    # convert to pandas series
    perf = pd.concat([pd.Series(perf_dict_base), 
                      pd.Series(perf_dict), 
                      pd.Series({k:v[0] for k,v in perf_dict_diff_p.items()}),
                      pd.Series({k:v[1] for k,v in perf_dict_diff_p.items()})],
                      axis = 1,
                      keys = [(mdl_name, 'Base'), 
                              (mdl_name, '+Fitbit'), 
                              (mdl_name, 'Difference (95% CI)'),
                              (mdl_name, 'p-value')])

    ##### plotting (for full model only) #####
    # 1. Probability distribution stratified by true label
    probs_pos_class = y_prob[y_true==1]
    probs_neg_class = y_prob[y_true==0]
    ax1.hist(probs_pos_class, bins=30, alpha=0.6, label='Positive (True=1)', color='blue', density=True)
    ax1.hist(probs_neg_class, bins=30, alpha=0.6, label='Negative (True=0)', color='red', density=True)
    ax1.legend(loc = "lower right")
    ax1.set_title(mdl_name)

    return perf, ax1

def get_optimal_threshold(y, probs):
    precision, recall, thresholds = precision_recall_curve(y, probs)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-12)

    best_idx = np.argmax(f1)
    best_threshold = thresholds[best_idx]
    return best_threshold

def create_logreg_mdl(X_train, X_test, y_train, y_test,
                      subcohort, features, feature_rename_map, output_folder, 
                      alpha = 0.05,
                      scale_pos_weight = False):

    # compute event per variabale
    n_events = min(len(y_train) - y_train.sum(), y_train.sum())
    epv = n_events/len(features)
    print(f'Event per variable = {int(epv)}, N events {n_events} / degree of freedom {len(features)}')

    # frequency weights for class imbalance
    if scale_pos_weight:
        pos_weight = (y_train == 0).sum()/(y_train == 1).sum()
        freq_weights = np.where(y_train==1, pos_weight, 1)
        clf = sm.GLM(y_train, X_train, family=sm.families.Binomial(), freq_weights = freq_weights)
    else:
        clf = sm.Logit(y_train, X_train)
    results = clf.fit(maxiter = 350)

    # Get the optimal threshold using training data only
    optimal_threshold = get_optimal_threshold(y_train, results.predict(X_train))

    y_proba = results.predict(X_test)
    test_auc = roc_auc_score(y_test, y_proba)
    y_pred = (y_proba > 0.5).astype(int)
    # print(f'Pseudo R squared: {results.prsquared: .2f}')
    print(f'Test ROC AUC: {test_auc: .2f}')
    print(classification_report(y_test, y_pred))
    # get summary df and print the variables with significant p value
    summary = get_logreg_results(results, alpha = alpha).sort_values('p')
    # print(summary.loc[summary['p']<=0.05])
    # add only the significant variables to list of summaries
    df_fmt, idx_map, latex = format_reg_results(summary, subcohort.replace('subcohort_', ''), 
                                                rename_map = feature_rename_map, multiindex=True,
                                                alpha = alpha)
    df_fmt = df_fmt[['Coef', r'\textit{p}', r'\textit{Adjusted p}']]
    # amend the index
    df_fmt.columns = pd.MultiIndex.from_tuples([(subcohort.replace('subcohort_', ''), x) for x in['Coef', r'\textit{p}', r'\textit{Adjusted p}']])

    with open(f"{output_folder}{subcohort}.tex", 'w') as f:
        f.write(latex)
    
    return df_fmt, idx_map, y_proba, optimal_threshold


def create_mdl_xgb(X_train, y_train, X_test, 
                   subcohort,
                   savefolder, 
                   imb_data = False,
                   scale_pos_weight = False, 
                   rename_map = None, 
                   multiindex=False):

    if imb_data:
        print('Applying SMOTE to balance the two classes...')
        ros = RandomOverSampler(random_state=42)
        X_train, y_train = ros.fit_resample(X_train, y_train) 

    # Make sure the number of threads is balanced.
    if scale_pos_weight:
        xgb_model = xgb.XGBClassifier(
            n_jobs = multiprocessing.cpu_count() // 2, 
            tree_method = "hist",
            enable_categorical = True,
            scale_pos_weight = (y_train == 0).sum()/(y_train == 1).sum()
        )
    else:
        xgb_model = xgb.XGBClassifier(
            n_jobs = multiprocessing.cpu_count() // 2, 
            tree_method = "hist",
            enable_categorical = True
        )
    clf = GridSearchCV(
        xgb_model,
        {"max_depth": [2, 4, 6], "n_estimators": [50, 100, 200]},
        verbose = 1,
        n_jobs = 2,
        
    )

    clf.fit(X_train, y_train)

    print(clf.best_score_)
    best_params = clf.best_params_
    print(best_params)
    # Get the best estimator
    clf = clf.best_estimator_

    # Get the optimal threshold using training data only
    optimal_threshold = get_optimal_threshold(y_train, clf.predict_proba(X_train)[:,1])

    y_proba = clf.predict_proba(X_test)[:,1]

    ### evaluate the stability of performance and shap importance across CV folds (train set) ###
    impt_cv, perf_cv = eval_xgb_cv(X_train, y_train, best_params, n_splits = 5)
    ### final feature importance and shap values in test set using the clf fit on entire train set ###
    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(X_test)
    # compute mean abs and signed shap values
    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    mean_shap = np.mean(shap_vals, axis=0)
    impt = pd.DataFrame({
        "feature": X_train.columns,
        "mean_abs_shap_test": mean_abs,
        'mean_signed_shap_test': mean_shap}).sort_values('mean_abs_shap_test', ascending=False)
    # merge with the shap values from cv folds
    impt = impt.merge(impt_cv, on = 'feature')
    impt = impt.set_index('feature')

     # Optional: nicer variable names, multiindex to include
    if rename_map is not None:
        print(impt)
        impt, _ = rename_index(impt, rename_map, multiindex=multiindex)
    
    # save importance
    impt.to_csv(f"{savefolder}{subcohort}.csv", index=True)
    # save as latex
    impt.to_latex(
        f"{savefolder}{subcohort}.tex",
        escape = False,
        float_format="%.3f",
        column_format="lrrrrr",
        caption=f"Absolute mean SHAP values of subcohort {subcohort} with 95\% confidence interval"
    )

    return clf, y_proba, impt, perf_cv, optimal_threshold

def eval_xgb_cv(X_train, y_train, params, n_splits = 5):
    """
    Evaluate the stability of """
    skf = StratifiedKFold(n_splits = n_splits, shuffle=True, random_state=42)


    auc_scores = []
    bss = []
    fold_importances = []
    fold_ranks = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr)

        # --- Performance ---
        y_pred = model.predict_proba(X_val)[:, 1]
        auc_scores.append(roc_auc_score(y_val, y_pred))
        bss.append(scaled_brier_score(y_val, y_pred))

        # --- SHAP on held-out fold ---
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_val)

        mean_abs = np.mean(np.abs(shap_vals), axis=0)
        fold_importances.append(mean_abs)

        fold_ranks.append(
            pd.Series(mean_abs, index=X_train.columns).rank(ascending=False)
        )
    
    ###### evaluate stability of such metrics ######
    # Performance
    cv_auc_mean = np.mean(auc_scores)
    cv_auc_sd = np.std(auc_scores)
    cv_bss_mean = np.mean(bss)
    cv_bss_sd = np.std(bss)

    # Importance magnitude stability
    importances = np.vstack(fold_importances)

    mean_importance = np.mean(importances, axis=0)
    sd_importance = np.std(importances, axis=0)
    cv_importance = sd_importance / mean_importance  # coefficient of variation

    # Rank stability (Spearman)
    rank_corrs = []
    for i in range(n_splits):
        for j in range(i+1, n_splits):
            rho, _ = spearmanr(fold_ranks[i], fold_ranks[j])
            rank_corrs.append(rho)

    mean_rank_stability = np.mean(rank_corrs)

    impt = pd.DataFrame({
        "feature": X_train.columns,
        "mean_abs_shap_cv": mean_importance,
        "sd_abs_shap_cv": sd_importance,
        "cv_abs_shap_cv": cv_importance
    }).sort_values("mean_abs_shap_cv", ascending=False)

    results_dict = pd.Series({
        'cv_auc_mean': cv_auc_mean,
        'cv_auc_sd': cv_auc_sd,
        'cv_bss_mean': cv_bss_mean,
        'cv_bss_sd': cv_bss_sd,
        'mean_rank_stability': mean_rank_stability
    })

    print("CV AUROC:", cv_auc_mean, "+/-", cv_auc_sd)
    print("Mean Rank Stability (Spearman):", mean_rank_stability)

    return impt, results_dict


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
                config, subcohort_cols, 
                train_dataloader, val_dataloader, test_dataloader,
                mean_features, std_features,
                mean_static, std_static,
                pos_weight,
                results_folder,
                with_static = False):

    ### Read conf
    max_epochs = config['max_epoch']
    lr = float(config['initial_lr'])
    weight_decay = config['weight_decay']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training using {device}")

    ######## Optimizer and learning rate scheduler ########
    optimizer = optim.AdamW(itertools.chain(model.parameters(),
                                            clf.parameters()), 
                            lr = lr, 
                            weight_decay = weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0 = config['cosine_anneal_lr_scheduler']['T_0'],
        T_mult = config['cosine_anneal_lr_scheduler']['T_mult'],
        eta_min = float(config['cosine_anneal_lr_scheduler']['eta_min'])
    )

    # Move model to device
    model = model.to(device)
    clf = clf.to(device)
    # define loss function, positive weight proportional to subcohort sizes relative to control 
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))
    # loss_fn = nn.BCEWithLogitsLoss()

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

            ##### Forward pass with transformer #####
            out = model(data)
            # Fuse with static data
            if with_static:
                out = torch.concat((out, data_static), dim = 1)
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
                # Fuse with static data
                if with_static:
                    out = torch.concat((out, data_static), dim = 1)
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
                                            with_static = with_static)

    for i in range(len(subcohort_cols)):
        subcohort = subcohort_cols[i]
        print(subcohort)
        eval_folder = f'{results_folder}/{subcohort}/'
        if not os.path.exists(eval_folder):
            os.makedirs(eval_folder)
        print(eval_folder)
        assess_clf_perf(test_pred[:,i], 
                        test_proba[:,i], 
                        test_labels[:,i], 
                        eval_folder,
                        labels = ['Negative (0)', 'Positive (1)'])


    return model, clf

def get_logreg_results(results, alpha = 0.05):
    df = results.conf_int().rename(columns = {0: 'CI_lower', 1: 'CI_upper'})
    df['p'] = results.pvalues
    _, df['p_adjusted'], _, _ = multipletests(df['p'], alpha = alpha, method = "fdr_bh")
    df['coef'] = np.round(results.params, 3)
    df = df[['coef', 'CI_lower', 'CI_upper', 'p', 'p_adjusted']]
    df['CI_lower'] = np.round(df['CI_lower'], 3)
    df['CI_upper'] = np.round(df['CI_upper'], 3)
    return df


def rename_index(df, rename_map, multiindex=False):
    if multiindex:
        idx_map = get_mapped_multiindex(rename_map, df.index)
    else:
        idx_map = {k: v for k, v in rename_map.items() if k in df.index} # simply filter using existing index
    # Sort using the order of the new indices
    # print(list(idx_map.keys()))
    df = df.loc[list(idx_map.keys())]
    # set index
    if multiindex:
        df.index = pd.MultiIndex.from_tuples(list(idx_map.values()))
    else:
        df.index = idx_map.values
    
    return df, idx_map

def get_mapped_multiindex(feature_rename_map, original_indices):
    """new index will follow the order in feature_rename_map"""
    multi_idx_rename_map = {}
    for group, rename_map in feature_rename_map.items():
        multi_idx_rename_map.update({ori_name: (group, new_name) for ori_name, new_name in rename_map.items()})
    return {k: v for k, v in multi_idx_rename_map.items() if k in original_indices}

def format_reg_results(df, subcohort, rename_map = None, multiindex = False, alpha = 0.05):

    df_fmt = df.copy()

    # Round numeric columns
    for col in ["CI_lower", "CI_upper"]:
        df_fmt[col] = df_fmt[col].round(3)
    
    df_fmt['95%_CI'] = df_fmt.apply(lambda x: f"{x['CI_lower']} - {x['CI_upper']}", axis=1)

    # Optional: nicer variable names, multiindex to include
    if rename_map is not None:
        df_fmt, idx_map = rename_index(df_fmt, rename_map, multiindex=multiindex)
        
    # # save the variables that are significant
    # df_fmt_sig_idx = df_fmt.loc[df_fmt['p'] <= alpha].index

    # Format p-values
    df_fmt["p"] = df_fmt["p"].apply(format_p)
    df_fmt["p_adjusted"] = df_fmt["p_adjusted"].apply(format_p)

    # Rename columns for publication
    df_fmt = df_fmt[['coef', '95%_CI', 'p', 'p_adjusted']].rename(
        columns={
            "coef": "Coef",
            '95%_CI': "95\% CI",
            "p": r"\textit{p}",
            "p_adjusted": r"\textit{Adjusted p}",
        }
    )

    # -----------------------------
    # Export to LaTeX
    # -----------------------------
    latex_table = df_fmt.to_latex(
        escape=False,       # allow LaTeX formatting (e.g. \textit{p})
        float_format="%.3f",
        column_format="lrrrrr",
        caption=f"Regression coefficients of subcohort {subcohort} with 95\% confidence interval",
        label="tab:regression",
    )

    return df_fmt, idx_map, latex_table


def assign_train_test(indices, test_size = 0.3, seed = 42):
    rng = np.random.default_rng(seed)

    rng.shuffle(indices)
    # compute test size
    test_size = int(len(indices) * test_size)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]

    return train_idx, test_idx


def make_mdl_df_static(cohort, 
                       survey_df, 
                       time_series_sumstats,
                       time_series_features,
                       rmssd_features,
                       subcohort_cols,
                       static_features,
                       survey_features,
                       excl_vars_ts, # time series variables to be excluded
                       excl_vars_static, # static features to be excluded
                    #    ordinal_var_mapping,
                       test_size = 0.3,
                       baseline_only = False):


    ######## get features #######
    sumstats_features = [f"{x}_{suffix}" for x in time_series_features for suffix in ['mean', 'std']] + \
                        [f"{x}_rmssd" for x in rmssd_features]
    features = static_features + survey_features + sumstats_features
    # exclude some features; the ones excluded are selected in a separate notebook based on correlation
    # and domain knowledge
    excl_vars = [f'{x}_{suffix}' for x in excl_vars_ts for suffix in ['mean', 'std', 'rmssd']] + excl_vars_static
    features = [x for x in features if x not in excl_vars]

    # Create model df
    df = cohort.merge(time_series_sumstats).merge(survey_df, how = 'left')
    # # change std to cv
    # for feat in time_series_features:
    #     df[f'{feat}_cv'] = df[f'{feat}_std']/df[f'{feat}_mean']

    # # some variables need to be converted from ordinal back to binary variables
    # for var, mapping in ordinal_var_mapping.items():
    #     var_df = pd.get_dummies(df[[var]].dropna().astype(int).map(lambda x: mapping.get(x))).astype(int)
    #     df = df.join(var_df, how = 'inner')
    #     # modify features
    #     features = [x for x in features if x != var] + list(var_df.columns)
    
    df.to_csv('data/temp/mdl_df_temp.csv', index=False)
    df = df[['person_id', 'index_date'] + subcohort_cols + features].dropna(subset = features)

    # change stpes to units of 1000 for easier interpretation
    if 'steps_mean' in df.columns:
        df['steps_mean'] = df['steps_mean']/1000

    # only baseline_only is true, don't use the sumstats features
    if baseline_only:
        features = [x for x in features if x not in sumstats_features]
        df = df[['person_id', 'index_date'] + subcohort_cols + features]

    # split into train and test set manually
    train_idx, test_idx = assign_train_test(df.index.tolist(), test_size = test_size)
    df.loc[train_idx, 'train_test_label'] = 'train'
    df.loc[test_idx, 'train_test_label'] = 'test'
    
    return df, features

def create_comp_df_xgb(xgb_res, subcohort_cols, output_folder):

    # cross validated performance metrics
    cv_perf = zip(subcohort_cols, [x[1] for x in xgb_res])
    cv_perf = pd.DataFrame(dict(cv_perf))
    cv_perf.to_csv(f"{output_folder}cv_perf.csv")

    # re-format and save as latex
    cv_perf = cv_perf.transpose()
    cv_perf['ROC AUC'] = cv_perf.apply(lambda df: 
                f"${float(df['cv_auc_mean']):.3f} \pm {float(df['cv_auc_sd']):.3f}$", axis=1)
    cv_perf['BSS'] = cv_perf.apply(lambda df: 
                f"${float(df['cv_bss_mean']):.3f} \pm {float(df['cv_bss_sd']):.3f}$", axis=1)
    cv_perf['Mean rank stability'] = cv_perf['mean_rank_stability']
    cv_perf.index = [x.replace('subcohort_', '') for x in cv_perf.index]
    latex_str = cv_perf[['ROC AUC', 'BSS', 'Mean rank stability']].to_latex(
                None,
                escape = False,
                float_format = "%.3f",
                caption = 'Performance stability of XGB model using 5-fold CV',
                label = 'tab:xgb_cv_perf'
            )
    with open(f"{output_folder}cv_perf.tex", 'w') as f:
        f.write(latex_str)

    # absolute mean shap comparison
    shap_comps = zip(subcohort_cols, [x[0] for x in xgb_res])
    abs_shaps = []
    for sub, shap_sub in shap_comps:
        shap_sub = shap_sub[['mean_abs_shap_test']].rename(
            columns = {'mean_abs_shap_test': sub.replace('subcohort_', '')}
        )
        abs_shaps.append(shap_sub)
    abs_shaps = reduce(lambda x, y: x.join(y), abs_shaps)
    print(abs_shaps)

    # save to latex
    abs_shaps.to_latex(
        f"{output_folder}abs_mean_shap.tex",
        longtable = True,
        escape = False,
        float_format = "%.3f",
        caption = 'Absolute Mean SHAP values across subcohorts on test dataset',
        label = 'tab:xgb_abs_mean_shap'
    )


    return



def create_comp_df_logreg(logreg_res, idx_map, subcohort_cols, output_folder):

    # Compare results across all cohorts for logreg
    comp_df = reduce(lambda x, y: x.join(y, how = 'outer'), logreg_res)
    # Select those variables significant in at least one subcohort
    comp_df = comp_df.loc[comp_df[[(x.replace('subcohort_', ''), r'\textit{Adjusted p}') for x in subcohort_cols]]\
                            .apply(lambda x: x.str.contains('\*', regex=True).any(), axis=1)]
    # order the index
    comp_df = comp_df.loc[[x for x in list(idx_map.values()) if x in comp_df.index]]
    # drop nominal p-value and only keep adjusted p value
    comp_df = comp_df.drop(columns = r'\textit{p}', level = 1)

    # export to latex
    comp_latex = comp_df.to_latex(
        None,
        escape=False,       # allow LaTeX formatting (e.g. \textit{p})
        float_format="%.3f",
        longtable = True,
        bold_rows = True,
        multicolumn_format = 'c',
        column_format = r"l>{\raggedright\arraybackslash\setstretch{0.85}}p{2.8cm}rlrlrlrlrlrlrl",
        caption= r"Regression coefficients across subcohorts at 5\% significance level with Benjamini–Hochberg (BH) adjusted p-values",
        label="tab:regression",
    )

    # comp_latex = comp_latex.replace(r'\textbackslash textit\{p\}', r'\textit{p}')
    with open(f'{output_folder}/comp.tex', 'w') as f:
        f.write(r"""\begin{landscape}
                    \footnotesize
                    \setlength{\tabcolsep}{3pt}
                """)
        f.write(comp_latex)
        f.write(r"\end{landscape}")
    return


def create_metric_comp(logreg_metrics, xgb_metrics,
                       logreg_xgb_metrics_comp,
                    subcohort_cols, output_folder):

    def concat_helper(metrics):
        metrics = pd.concat(metrics, axis = 1)
        return metrics
    # Concatenate everything
    logreg_metrics = concat_helper(logreg_metrics)
    xgb_metrics = concat_helper(xgb_metrics)
    logreg_xgb_metrics_comp = concat_helper(logreg_xgb_metrics_comp)
    print(logreg_xgb_metrics_comp)

    logreg_metrics.to_csv(f"{output_folder}/logreg_metrics.csv", index=True)
    xgb_metrics.to_csv(f"{output_folder}/xgb_metrics.csv", index=True)
    logreg_xgb_metrics_comp.to_csv(f"{output_folder}/logreg_xgb_metrics_comp.csv", index=True)

    # Combine and index them using multiindex
    df = pd.concat([logreg_metrics, xgb_metrics, logreg_xgb_metrics_comp], 
                   axis = 1, 
                   keys = ['LogReg', 
                            'XGBoost',
                            'XGB vs LogReg Difference'
                            ])
    # ordering
    df = df.swaplevel(0, 1, axis=1)

    print(df)
    df.to_csv(f'{output_folder}/static_mdl_perf.csv', index = True)

    ####### Stack the df and output all metrics to latex #######
    idx_ordered = list(tuple(itertools.product(
                            [x.replace('subcohort_', '') for x in subcohort_cols], 
                            ['Accuracy',
                            'Precision',
                            'Recall',
                            'F1 Score',
                            'ROC AUC Score',
                            'Brier Score',
                            'Brier Skill Score (BSS)'])))
    df_latex = df.stack(level=0).swaplevel(0 , 1, axis=0).sort_index(level=0).loc[idx_ordered].copy()

    latex_str = df_latex.to_latex(None,
                            escape=False,
                            float_format="%.3f",
                            longtable = True,
                            bold_rows = True,
                            multicolumn_format = 'c',
                            caption = 'Logistic regression and XGBoost model performance across subcohorts (95\% CI)',
                            label = 'tab:clfmetrics'
                        )
    
    latex_str = r'\begin{landscape}\footnotesize' + latex_str + r'\end{landscape}'
    latex_str = latex_str.replace(r"(95% CI)", r"(95\% CI)")

    
    with open(f'{output_folder}/static_mdl_perf.tex', 'w') as f:
        f.write(latex_str)


    ####### Output just the key performance metrics with p-value #######
    core_perf = df.swaplevel(0, 1, axis=1)[['LogReg', 'XGBoost']].swaplevel(0, 1, axis=1)\
                .swaplevel(0, 2, axis=1).loc[['ROC AUC Score', 'Brier Skill Score (BSS)'], \
                ['Difference (95% CI)', 'p-value']].swaplevel(0, 2, axis=1).transpose().unstack(2)
    core_perf = core_perf.loc[[x.replace('subcohort_', '') for x in subcohort_cols]]
    core_perf.to_csv(f'{output_folder}/core_static_mdl_perf.csv')

    core_perf_latex_str = core_perf.to_latex(None,
                            escape=False,
                            float_format="%.3f",
                            longtable = True,
                            bold_rows = True,
                            multicolumn_format = 'c',
                            caption = 'Improvement in Discrimination (AUROC) and Calibration (BSS) after incorporating FitBit features (95\% CI)',
                            label = 'tab:coreclfmetrics'
                        )
    core_perf_latex_str = core_perf_latex_str.replace(r"(95% CI)", r"(95\% CI)")
    with open(f'{output_folder}/core_static_mdl_perf.tex', 'w') as f:
        f.write(core_perf_latex_str)
    
    ####### Output just xgb vs logreg difference #######
    core_diff = df.swaplevel(0, 1, axis=1)[['XGB vs LogReg Difference']].swaplevel(0, 1, axis=1)\
                .swaplevel(0, 2, axis=1).loc[['ROC AUC Score', 'Brier Skill Score (BSS)'], \
                ['Difference (95% CI)', 'p-value']].swaplevel(0, 2, axis=1).transpose().unstack(2)
    # order the index
    core_diff = core_diff.droplevel(1).loc[[x.replace('subcohort_', '') for x in subcohort_cols]]
    print(core_diff)
    core_diff.to_csv(f'{output_folder}/core_xgb_logreg_diff.csv')

    core_diff_latex_str = core_diff.to_latex(None,
                            escape=False,
                            float_format="%.3f",
                            longtable = True,
                            bold_rows = True,
                            multicolumn_format = 'c',
                            caption = 'Difference in performance between XGBoost and Logistic Regression model (95\% CI)',
                            label = 'tab:coreclfdiff'
                        )
    core_diff_latex_str = core_diff_latex_str.replace(r"(95% CI)", r"(95\% CI)")
    with open(f'{output_folder}/core_xgb_logreg_diff.tex', 'w') as f:
        f.write(core_diff_latex_str)


    ####### Plotting just ROC AUC and BSS to visualise key metrics #######
    df = df.swaplevel(0, 1, axis=1)[['LogReg', 'XGBoost']].swaplevel(0, 1, axis=1)\
        .loc[['ROC AUC Score', 'Brier Skill Score (BSS)']]\
        .stack(level=0).swaplevel(0 , 1, axis=0).sort_index(level=0).unstack(level=1)\
        .map(lambda x: list(map(float, re.findall(r"-?\d*\.?\d+", x)))) # separate out confidence intervals
    
    print(df)
    
    plot_key_clf_metrics(df, [x.replace('subcohort_', '') for x in subcohort_cols],
                         f'{output_folder}/static_mdl_perf_plt.tiff')
    
    return



def plot_key_clf_metrics(df, subcohorts, savepath):
    # ---- plotting setup ----
    subcohorts = subcohorts[::-1]  # reverse so first is on top
    df = df.loc[subcohorts]
    models = [('LogReg', 'Base'),
            ('LogReg', '+Fitbit'),
            ('XGBoost', 'Base'),
            ('XGBoost', '+Fitbit')]

    y_base = np.arange(len(subcohorts))
    offset = 0.2  # vertical offset between models

    model_offsets = {
        ('LogReg', 'Base'): -1.5*offset,
        ('LogReg', '+Fitbit'): -offset/2,
        ('XGBoost', 'Base'): offset/2,
        ('XGBoost', '+Fitbit'): 1.5*offset
    }

    model_formats = {
        ('LogReg', 'Base'): {
            'color': "#6BAED6",
            'fmt': 'o',
            'alpha': 1
        },
        ('LogReg', '+Fitbit'): {
            'color': "tab:blue",
            'fmt': 'o',
            'alpha': 1
        },
        ('XGBoost', 'Base'): {
            'color': "#FDAE6B",
            'fmt': 'o',
            'alpha': 1
        },
        ('XGBoost', '+Fitbit'): {
            'color': "tab:orange",
            'fmt': 'o',
            'alpha': 1
        },
    }

    fig, axs = plt.subplots(
        ncols=2,
        sharey=True,
        figsize=(12, 7),
        gridspec_kw={"wspace": 0.25}
    )


    # ---- Plotting for ROC AUC and BSS ----
    i = 0
    for metric in ['ROC AUC Score', 'Brier Skill Score (BSS)']:
        for model in models:
            y = y_base + model_offsets[model]

            scores = np.array([x[0] for x in df[model][metric]])
            ci_lower = np.array([x[1] for x in df[model][metric]])
            ci_higher = np.array([x[2] for x in df[model][metric]])

            axs[i].errorbar(
                scores,
                y,
                xerr=[
                    scores - ci_lower,
                    ci_higher - scores,
                ],
                capsize=3,
                label=model,
                **model_formats[model]
            )
            pad = 0.005

            text_labels = [f"{float(x[0]):.2f} ({float(x[1]):.2f} - {float(x[2]):.2f})" for x in df[model][metric]]
            for x_pos, y_pos, text_label in zip(ci_higher + pad, y, text_labels):
                axs[i].text(
                    x_pos,
                    y_pos,
                    text_label,
                    va="center",
                    ha="left",
                    fontsize=9
                )

        # axs[i].axvline(0.0, linestyle="--", linewidth=1)
        axs[i].set_xlabel(metric, fontsize = 14)
        axs[i].set_title(metric, fontsize = 15)

        # set sizes for x ticks
        axs[i].tick_params(axis='x', labelsize=14)

        i += 1

    # ---- shared y-axis ----
    axs[0].set_yticks(y_base)
    axs[0].set_yticklabels(subcohorts)
    # set sizes for s ticks
    axs[0].tick_params(axis='y', labelsize=14)

    # ---- legend & cleanup ----
    axs[0].legend(frameon=False, title="Model", fontsize = 14)

    for ax in axs:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(savepath, dpi = 150, pil_kwargs={"compression": "tiff_lzw"})

    return


def create_logreg_and_xgb_mdl_comp(cohort, 
                                    survey_df, 
                                    time_series_sumstats,
                                    time_series_features,
                                    rmssd_features,
                                    subcohort_cols,
                                    static_features,
                                    survey_features,
                                    excl_vars_ts, # time series variables to be excluded
                                    excl_vars_static, # static features to be excluded
                                    # ordinal_var_mapping,
                                    feature_rename_map,
                                    results_folder_logreg,
                                    results_folder_xgb,
                                    results_folder_logreg_base,
                                    results_folder_xgb_base,
                                    fig1_path,
                                    fig2_path,
                                    fig3_path,
                                    test_size = 0.3,
                                    scale_pos_weight = False):
        # get model df
        df, features = make_mdl_df_static(cohort, 
                                            survey_df, 
                                            time_series_sumstats,
                                            time_series_features,
                                            rmssd_features,
                                            subcohort_cols,
                                            static_features,
                                            survey_features,
                                            excl_vars_ts, # time series variables to be excluded
                                            excl_vars_static, # statics features to be excluded
                                            # ordinal_var_mapping,
                                            test_size = test_size)
        
        df.to_csv('data/temp/cohort_sumstats_20260225.csv', index=False)
        print(df)

        logreg_res = [] # df for aggregating logreg results (coefficients)
        logreg_metrics = [] # for aggregating logreg metrics
        xgb_res = [] # for aggregating xgb results
        xgb_metrics = [] # for aggregating xgb metrics
        logreg_xgb_metrics_comp = [] # compare diff between logreg and xgb
        # Initialise figure to plot everything on the same chart
        fig1, axs1 = plt.subplots(len(subcohort_cols), 2, figsize=(8, 22))
        fig2, axs2 = plt.subplots(3, 3, figsize=(12, 12))
        fig3, axs3 = plt.subplots(3, 3, figsize=(12, 12))
        # Keep a list to track which axes have been used to plot
        used_indices = []

        for subcohort in subcohort_cols:
            print(subcohort)
            i = subcohort_cols.index(subcohort)

            # get train test dataset
            mdl_df = pd.concat([df.loc[df[subcohort]==1], 
                            df.loc[(df[subcohort_cols]==0).all(axis=1)]])
            # split into train and test
            df_train = mdl_df.loc[mdl_df['train_test_label'] == 'train']
            df_test = mdl_df.loc[mdl_df['train_test_label'] == 'test']

            # features just baseline ones without fitbit
            features_base = [x for x in features if x not in time_series_sumstats]

            # Get train and test
            X_train = df_train[features]
            X_train_base = df_train[features_base]
            y_train = df_train[subcohort]
            X_test = df_test[features]
            X_test_base = df_test[features_base]
            y_test = df_test[subcohort]

            ########################
            # Logistic regression
            ########################
            # LogReg model full
            logreg_res_fmt, idx_map, y_proba_logreg, threshold_logreg = create_logreg_mdl(sm.add_constant(X_train), 
                                                                                        sm.add_constant(X_test), 
                                                                                        y_train, 
                                                                                        y_test,
                                                                                        subcohort, 
                                                                                        features, 
                                                                                        feature_rename_map,
                                                                                        results_folder_logreg,
                                                                                        scale_pos_weight = scale_pos_weight) 
            # LogReg model base - note that we don't keep the coefficients
            _, _, y_proba_logreg_base, threshold_logreg_base = create_logreg_mdl(sm.add_constant(X_train_base), 
                                                                                sm.add_constant(X_test_base), 
                                                                                y_train, 
                                                                                y_test,
                                                                                subcohort, 
                                                                                features_base, 
                                                                                feature_rename_map,
                                                                                results_folder_logreg_base,
                                                                                scale_pos_weight = scale_pos_weight) 
            ### Get LogReg model metrics
            logreg_res.append(logreg_res_fmt)
            logreg_metric, axs1[i, 0] = get_clf_metrics(y_test, y_proba_logreg_base,
                                                        y_proba_logreg, 
                                                        f"{subcohort.replace('subcohort_', '')}", 
                                                        axs1[i, 0],
                                                        threshold_logreg_base, threshold_logreg)
            # print(logreg_metric)
            # logreg_metric.to_csv('results/temp/logreg_metric.csv')
            logreg_metrics.append(logreg_metric)
            
            ########################
            # XGBoost model
            ########################
            # XGB model full
            clf, y_proba_xgb, impt, perf_cv, threshold_xgb = create_mdl_xgb(X_train, y_train, X_test, 
                                                                            subcohort,
                                                                            results_folder_xgb, 
                                                                            imb_data = False,
                                                                            scale_pos_weight = scale_pos_weight,
                                                                            rename_map = feature_rename_map,
                                                                            multiindex = True)
            assess_clf_perf_xgb(clf, X_test, y_test, 
                                f'{results_folder_xgb}/{subcohort}/',
                                features,
                                feature_rename_map = None,
                                labels = ['Negative (0)', 'Positive (1)'])
            ### XGB model base
            clf, y_proba_xgb_base, _, _, threshold_xgb_base = create_mdl_xgb(X_train_base, y_train, X_test_base,
                                                                            subcohort,
                                                                            results_folder_xgb_base,
                                                                            imb_data = False,
                                                                            scale_pos_weight = scale_pos_weight, 
                                                                            rename_map = feature_rename_map,
                                                                            multiindex = True)
            assess_clf_perf_xgb(clf, X_test_base, y_test, 
                                f'{results_folder_xgb_base}/{subcohort}/',
                                features_base,
                                feature_rename_map = None,
                                labels = ['Negative (0)', 'Positive (1)'])
            ### Get XGB model metrics
            xgb_metric, axs1[i, 1] = get_clf_metrics(y_test, y_proba_xgb_base, y_proba_xgb, 
                                                    f"{subcohort.replace('subcohort_', '')}", 
                                                    axs1[i, 1],
                                                    threshold_xgb_base, threshold_xgb)
            xgb_metrics.append(xgb_metric)
            xgb_res.append((impt, perf_cv))
            print(xgb_res)

            # compare performance of xgb against logreg
            logreg_xgb_diff = comp_perf_logreg_xgb(y_test, y_proba_logreg, y_proba_xgb,
                                                f"{subcohort.replace('subcohort_', '')}",
                                                threshold_logreg, threshold_xgb)
            logreg_xgb_metrics_comp.append(logreg_xgb_diff)

            # Plot calibration curve and decision curve
            plot_calibration_curve(y_test, {'LogReg': y_proba_logreg,
                                        'XGBoost': y_proba_xgb}, 
                                axs2[int(i/3), i%3], title = subcohort.replace('subcohort_', 'Calibration Curve: '))
            plot_decision_curve(y_test, {'LogReg': y_proba_logreg,
                                        'XGBoost': y_proba_xgb}, 
                                axs3[int(i/3), i%3], title = subcohort.replace('subcohort_', 'Decision Curve: '))
            used_indices.append((int(i/3), i%3))  

        # turn off the axis for unused axes
        all_indices = list(itertools.product([0, 1, 2], [0, 1, 2]))
        print(used_indices)
        for idx in all_indices:
            if not idx in used_indices:
                axs2[idx[0], idx[1]].axis('off')
                axs3[idx[0], idx[1]].axis('off')

        ### comparison across subcohorts - LogReg coefficients
        create_comp_df_logreg(logreg_res, idx_map, subcohort_cols, results_folder_logreg)
        create_comp_df_xgb(xgb_res, subcohort_cols, results_folder_xgb)

        ### Save comparison figure
        fig1.tight_layout()
        fig2.tight_layout()
        fig3.tight_layout()
        fig1.savefig(fig1_path, dpi = 300)
        fig2.savefig(fig2_path, dpi = 150, pil_kwargs={"compression": "tiff_lzw"})
        fig3.savefig(fig3_path, dpi = 300)

        return logreg_metrics, xgb_metrics, logreg_xgb_metrics_comp


def main():

    plt.rcParams.update({
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 13,
        'axes.labelsize': 14
    })
    config = yaml.safe_load(open('conf/base_mdl_fitbit.yaml'))
    scale_pos_weight = config['scale_pos_weight']
    ## read cohort and feature dataframes ##
    cohort = pd.read_csv(config['cohort'])
    subcohort_cols = config['subcohort_cols']
    time_series_feature_df = pd.read_csv(config['time_series_feature_df'])
    time_series_sumstats = pd.read_csv(config['time_series_sumstats']) # for XGBoost model
    time_series_features = config['time_series_features']
    rmssd_features = config['rmssd_features']
    survey_features = config['survey_features']
    survey_df = pd.read_csv(config['survey_df'])[['person_id']+survey_features].astype(int)
    static_features = config['static_features']
    feature_rename_map = config['feature_rename_map']
    results_folder = config['results_folder']
    mdl_conf = config['transformer_mdl_setting']
    # variables to be excluded
    excl_vars_ts = config['excl_vars_ts']
    excl_vars_static = config['excl_vars_static']
    # # mapping ordinal variables to dummy variables
    # ordinal_var_mapping = config['ordinal_var_mapping']

    # modify cohort so subcohort columns are 0 and 1
    cohort[subcohort_cols] = cohort[subcohort_cols].mask(cohort[subcohort_cols].notna(), 1)
    cohort[subcohort_cols] = cohort[subcohort_cols].fillna(0)

    # create results folder
    results_folder_ts_only = f'{results_folder}/ts_only/'
    results_folder_with_static = f'{results_folder}/with_static/'
    results_folder_xgb = f'{results_folder}/xgb/'
    results_folder_logreg = f'{results_folder}/logreg/'
    results_folder_xgb_base = f'{results_folder}/xgb_base/'
    results_folder_logreg_base = f'{results_folder}/logreg_base/'

    # ensure directory exists
    # if not os.path.exists(results_folder):
    #     os.makedirs(results_folder)
    if not os.path.exists(results_folder_ts_only):
        os.makedirs(results_folder_ts_only)
    if not os.path.exists(results_folder_with_static):
        os.makedirs(results_folder_with_static)
    if not os.path.exists(results_folder_xgb_base):
        os.makedirs(results_folder_xgb_base)
    if not os.path.exists(results_folder_logreg_base):
        os.makedirs(results_folder_logreg_base)
    if not os.path.exists(results_folder_xgb):
        os.makedirs(results_folder_xgb)
    if not os.path.exists(results_folder_logreg):
        os.makedirs(results_folder_logreg)


    ##################################
    ##### XGBoost models #####
    ##################################

    if config['xgb_run']:

        # Model with just baseline features
        logreg_metrics, xgb_metrics, logreg_xgb_metrics_comp = create_logreg_and_xgb_mdl_comp(cohort, 
                                                                    survey_df, 
                                                                    time_series_sumstats,
                                                                    time_series_features,
                                                                    rmssd_features,
                                                                    subcohort_cols,
                                                                    static_features,
                                                                    survey_features,
                                                                    excl_vars_ts, # time series variables to be excluded
                                                                    excl_vars_static, # static features to be excluded
                                                                    # ordinal_var_mapping,
                                                                    feature_rename_map,
                                                                    results_folder_logreg,
                                                                    results_folder_xgb,
                                                                    results_folder_logreg_base,
                                                                    results_folder_xgb_base,
                                                                    f'{results_folder}/static_mdl_proba_comp.png',
                                                                    f'{results_folder}/static_mdl_calibration_curve.tiff',
                                                                    f'{results_folder}/static_mdl_decision_curve.png',
                                                                    test_size = 0.3,
                                                                    scale_pos_weight = scale_pos_weight)

        create_metric_comp(logreg_metrics,
                           xgb_metrics,
                           logreg_xgb_metrics_comp,
                           subcohort_cols, 
                           results_folder)
        # save a copy of the config
        with open(f"{results_folder}config.yaml", "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
            
        sys.exit()

    ##################################
    ##### Transformer models #####
    ##################################

    if config['transformer_run']:
        # calculate the relative proportion of subcohort to use as positive weights
        pos_weight = []
        for col in subcohort_cols:
            pos_weight.append(cohort.loc[cohort[col]==0].shape[0]/cohort.loc[cohort[col]==1].shape[0])
        
        # get time series cohort for transformer model
        cohort_ts = cohort.merge(time_series_feature_df)

        dataset = SleepDataset(cohort_ts, 
                                features = time_series_features, 
                                static_features = [x for x in static_features + survey_features if x not in excl_vars_static],
                                date_col = 'date', 
                                label_col = subcohort_cols)
        # Split into dataloader
        g = torch.Generator().manual_seed(42)
        train_dataloader, val_dataloader, test_dataloader = get_train_val_test_dataloader(dataset, 
                                                                                        mdl_conf['val_split'], 
                                                                                        mdl_conf['test_split'], 
                                                                                        mdl_conf['batch_size'], 
                                                                                        g)
        # Compute mean and standard deviation across train set for normalisation later 
        mean_features, std_features = compute_mean_and_std_batch(train_dataloader)
        mean_static, std_static = compute_mean_and_std_static_batch(train_dataloader)

        ##### time series feature only model #####
        input_dim = next(iter(dataset))[0].shape[1]
        model, clf = init_mdl(mdl_conf, input_dim, subcohort_cols)
        model, clf = mdl_train_loop(model, clf,
                                    mdl_conf['ts_only_config'], subcohort_cols, 
                                    train_dataloader, val_dataloader, test_dataloader,
                                    mean_features, std_features,
                                    mean_static, std_static,
                                    pos_weight,
                                    results_folder_ts_only)

        # Choose the best epoch using the validation loss
        chosen_epoch = 10 # TODO change this to either manually input chosen epoch, or compute automatically
        checkpoint_sleep_only = torch.load(f'{results_folder_ts_only}/checkpoint.pth', weights_only=False)
        model.load_state_dict(checkpoint_sleep_only['model_state_dict'][chosen_epoch])

        ##### add in static features #####
        with_static_config = mdl_conf['with_static_config']
        # whether to freeze model weights and put model into eval mode if specified
        if with_static_config['freeze_mdl_weights']:
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
        # reinitialise classifier head with the correct dimension
        clf = ClassifierHead(d_model = mdl_conf['d_model']+len(static_features), out_dim = len(subcohort_cols))
        # TODO get the model with the correct epoch
        # training loop
        model, clf = mdl_train_loop(model, clf,
                                    mdl_conf['with_static_config'], subcohort_cols, 
                                    train_dataloader, val_dataloader, test_dataloader,
                                    mean_features, std_features,
                                    mean_static, std_static,
                                    pos_weight,
                                    results_folder_with_static,
                                    with_static = True)
        

if __name__ == '__main__':
    main()