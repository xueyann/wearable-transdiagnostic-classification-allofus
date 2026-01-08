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
import statsmodels.api as sm

from lib.transformer import SleepDataset, TransformerClassifier, ClassifierHead, \
    get_train_val_test_dataloader, compute_mean_and_std_batch, \
    compute_mean_and_std_static_batch, normalise_features, get_pred, \
    assess_clf_perf
from src.mdl_sleep import assess_clf_perf as assess_clf_perf_xgb
from imblearn.over_sampling import RandomOverSampler

# Modelling
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

def create_mdl_xgb(df, features, subcohort, subcohort_cols, imb_data = False):

    # get train test dataset
    mdl_df = pd.concat([df.loc[df[subcohort]==1], 
                        df.loc[(df[subcohort_cols]==0).all(axis=1)]])
    # split into train and test
    df_train = mdl_df.loc[mdl_df['train_test_label'] == 'train']
    df_test = mdl_df.loc[mdl_df['train_test_label'] == 'test']

    # Create X and y, train and test
    X_train = df_train[features]
    y_train = df_train[subcohort]
    X_test = df_test[features]
    y_test = df_test[subcohort]
    
    if imb_data:
        print('Applying SMOTE to balance the two classes...')
        ros = RandomOverSampler(random_state=42)
        X_train, y_train = ros.fit_resample(X_train, y_train) 

    # Make sure the number of threads is balanced.
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
    print(clf.best_params_)
    # Get the best estimator
    clf = clf.best_estimator_

    return clf, X_test, y_test

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
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5]).to(device))
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

def get_logreg_results(results):
    df = results.conf_int().rename(columns = {0: 'CI_lower', 1: 'CI_upper'})
    df['p'] = results.pvalues
    df['coef'] = np.round(results.params, 3)
    df = df[['coef', 'CI_lower', 'CI_upper', 'p']]
    df['CI_lower'] = np.round(df['CI_lower'], 3)
    df['CI_upper'] = np.round(df['CI_upper'], 3)
    return df

def get_mapped_multiindex(feature_rename_map, original_indices):
    """new index will follow the order in feature_rename_map"""
    multi_idx_rename_map = {}
    for group, rename_map in feature_rename_map.items():
        multi_idx_rename_map.update({ori_name: (group, new_name) for ori_name, new_name in rename_map.items()})
    return {k: v for k, v in multi_idx_rename_map.items() if k in original_indices}

def format_p(p, alpha = 0.05):
    if p < 0.001:
        return r"$<0.001*$"
    elif p <= alpha:
        return f"{p:.3f}*"
    else:
        return f"{p:.3f}"

def format_reg_results(df, subcohort, rename_map = None, multiindex = False, alpha = 0.05):

    df_fmt = df.copy()

    # Round numeric columns
    for col in ["CI_lower", "CI_upper"]:
        df_fmt[col] = df_fmt[col].round(3)
    
    df_fmt['95%_CI'] = df_fmt.apply(lambda x: f"{x['CI_lower']} - {x['CI_upper']}", axis=1)

    # Optional: nicer variable names, multiindex to include
    if rename_map is not None:
        # Get the dictionary mapping from old to new indices, sort df_fmt using the order specified in new indice mapping
        if multiindex:
            idx_map = get_mapped_multiindex(rename_map, df_fmt.index)
        else:
            idx_map = {k: v for k, v in rename_map.items() if k in df_fmt.index} # simply filter using existing index
        # Sort using the order of the new indices
        # print(list(idx_map.keys()))
        df_fmt = df_fmt.loc[list(idx_map.keys())]
        # set index
        if multiindex:
            df_fmt.index = pd.MultiIndex.from_tuples(list(idx_map.values()))
        else:
            df_fmt.index = idx_map.values
        
    # # save the variables that are significant
    # df_fmt_sig_idx = df_fmt.loc[df_fmt['p'] <= alpha].index

    # Format p-values
    df_fmt["p"] = df_fmt["p"].apply(format_p)

    # Rename columns for publication
    df_fmt = df_fmt[['coef', '95%_CI', 'p']].rename(
        columns={
            "coef": "Coef",
            '95%_CI': "95\% CI",
            "p": r"\textit{p}",
        }
    )

    # -----------------------------
    # Export to LaTeX
    # -----------------------------
    latex_table = df_fmt.to_latex(
        escape=False,       # allow LaTeX formatting (e.g. \textit{p})
        float_format="%.3f",
        column_format="lrrrr",
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
                       subcohort_cols,
                       static_features,
                       survey_features,
                       excl_vars_ts, # time series variables to be excluded
                       excl_vars_static, # static features to be excluded
                    #    ordinal_var_mapping,
                       test_size = 0.3):


    ######## get features #######
    sumstats_features = [f"{x}_{suffix}" for x in time_series_features for suffix in ['mean', 'std']]
    features = static_features + survey_features + sumstats_features
    # exclude some features; the ones excluded are selected in a separate notebook based on correlation
    # and domain knowledge
    excl_vars = [f'{x}_{suffix}' for x in excl_vars_ts for suffix in ['mean', 'std']] + excl_vars_static
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
    
    df = df[['person_id', 'index_date'] + subcohort_cols + features].dropna(subset = features)

    # change stpes to units of 1000 for easier interpretation
    df['steps_mean'] = df['steps_mean']/1000

    # split into train and test set manually
    train_idx, test_idx = assign_train_test(df.index.tolist(), test_size = test_size)
    df.loc[train_idx, 'train_test_label'] = 'train'
    df.loc[test_idx, 'train_test_label'] = 'test'
    
    return df, features


def create_logreg_mdl(df, subcohort, subcohort_cols, features, feature_rename_map, output_folder):


    # get train test dataset
    mdl_df = pd.concat([df.loc[df[subcohort]==1], 
                        df.loc[(df[subcohort_cols]==0).all(axis=1)]])
    # split into train and test
    df_train = mdl_df.loc[mdl_df['train_test_label'] == 'train']
    df_test = mdl_df.loc[mdl_df['train_test_label'] == 'test']
    X_train, y_train = sm.add_constant(df_train[features]), df_train[subcohort]
    X_test, y_test = sm.add_constant(df_test[features]), df_test[subcohort]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 41)
    # compute event per variabale
    n_events = min(len(y_train) - y_train.sum(), y_train.sum())
    epv = n_events/len(features)
    print(f'Event per variable = {int(epv)}, N events {n_events} / degree of freedom {len(features)}')

    clf = sm.Logit(y_train, X_train)
    results = clf.fit(maxiter = 350)
    y_proba = results.predict(X_test)
    test_auc = roc_auc_score(y_test, y_proba)
    y_pred = (y_proba > 0.5).astype(int)
    print(f'Pseudo R squared: {results.prsquared: .2f}')
    print(f'Test ROC AUC: {test_auc: .2f}')
    print(classification_report(y_test, y_pred))
    # get summary df and print the variables with significant p value
    summary = get_logreg_results(results).sort_values('p')
    # print(summary.loc[summary['p']<=0.05])
    # add only the significant variables to list of summaries
    df_fmt, idx_map, latex = format_reg_results(summary, subcohort.replace('subcohort_', ''), 
                                                rename_map = feature_rename_map, multiindex=True)
    df_fmt = df_fmt[['Coef', r'\textit{p}']]
    # amend the index
    df_fmt.columns = pd.MultiIndex.from_tuples([(subcohort.replace('subcohort_', ''), x) for x in['Coef', r'\textit{p}']])

    with open(f"{output_folder}{subcohort}.tex", 'w') as f:
        f.write(latex)
    
    return df_fmt, idx_map

def create_comp_df_logreg(logreg_res, idx_map, subcohort_cols, output_folder):
    # Compare results across all cohorts for logreg
    comp_df = reduce(lambda x, y: x.join(y, how = 'outer'), logreg_res)
    # Select those variables significant in at least one subcohort
    comp_df = comp_df.loc[comp_df[[(x.replace('subcohort_', ''), r'\textit{p}') for x in subcohort_cols]]\
                            .apply(lambda x: x.str.contains('\*', regex=True).any(), axis=1)]
    # order the index
    comp_df = comp_df.loc[[x for x in list(idx_map.values()) if x in comp_df.index]]

    # export to latex
    comp_latex = comp_df.to_latex(
        None,
        escape=False,       # allow LaTeX formatting (e.g. \textit{p})
        float_format="%.3f",
        longtable = True,
        bold_rows = True,
        multicolumn_format = 'c',
        caption= r"Regression coefficients across subcohorts at 5\% significance level",
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

def main():
    config = yaml.safe_load(open('conf/base_mdl_fitbit.yaml'))
    ## read cohort and feature dataframes ##
    cohort = pd.read_csv(config['cohort'])
    subcohort_cols = config['subcohort_cols']
    time_series_feature_df = pd.read_csv(config['time_series_feature_df'])
    time_series_sumstats = pd.read_csv(config['time_series_sumstats']) # for XGBoost model
    time_series_features = config['time_series_features']
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

    # ensure directory exists
    # if not os.path.exists(results_folder):
    #     os.makedirs(results_folder)
    if not os.path.exists(results_folder_ts_only):
        os.makedirs(results_folder_ts_only)
    if not os.path.exists(results_folder_with_static):
        os.makedirs(results_folder_with_static)
    if not os.path.exists(results_folder_xgb):
        os.makedirs(results_folder_xgb)
    if not os.path.exists(results_folder_logreg):
        os.makedirs(results_folder_logreg)


    ##################################
    ##### XGBoost models #####
    ##################################

    if config['xgb_run']:
        # get model df
        df, features = make_mdl_df_static(cohort, 
                                            survey_df, 
                                            time_series_sumstats,
                                            time_series_features,
                                            subcohort_cols,
                                            static_features,
                                            survey_features,
                                            excl_vars_ts, # time series variables to be excluded
                                            excl_vars_static, # static features to be excluded
                                            # ordinal_var_mapping,
                                            test_size = 0.3)
        # df.to_csv('data/temp/cohort_sumstats_20260108.csv', index=False)

        logreg_res = [] # df for aggregating logreg results
        for subcohort in subcohort_cols:
            print(subcohort)
            # Logistic regression
            logreg_res_fmt, idx_map = create_logreg_mdl(df, subcohort, subcohort_cols, features, 
                                                        feature_rename_map,
                                                        results_folder_logreg)   
            logreg_res.append(logreg_res_fmt)      
            # XGBoost model and assess xgboost performance
            clf, X_test, y_test = create_mdl_xgb(df, features, subcohort, subcohort_cols, imb_data = False)
            assess_clf_perf_xgb(clf, X_test, y_test, 
                                f'{results_folder_xgb}/{subcohort}/',
                                features,
                                feature_rename_map = None,
                                labels = ['Negative (0)', 'Positive (1)'])
        create_comp_df_logreg(logreg_res, idx_map, subcohort_cols, results_folder_logreg)
    
            
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