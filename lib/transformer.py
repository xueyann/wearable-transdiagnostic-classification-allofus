import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import itertools
import torch
import torch.nn as nn
import importlib
import src.rl_hier_contrastive
importlib.reload(src.rl_hier_contrastive)
from src.rl_hier_contrastive import TransformerEncoder
from src.rl_hier_contrastive import assess_clf_perf as transformer_assess_clf_perf
from torch.utils.data import Dataset, Subset, DataLoader, random_split, WeightedRandomSampler
import torch.optim as optim
import torch.nn.functional as F

# Assessing model performance
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

class SleepDataset(torch.utils.data.Dataset):
    # Note: ensure that df does not contain categorical columns before passing in
    def __init__(self, df, id_col = 'person_id', max_len = 21,
                 features = ['sleep_onset_delay_min', 'total_sleep_time_min',
                            'total_time_in_bed_min', 'sleep_efficiency_percent', 'waso_min',
                            'num_awakenings', 'fragmentation_index',
                            'longest_uninterrupted_sleep_min', 'mean_sleep_bout_duration_in_min',
                            'rem_latency_min', 'rem_fragmentation', 'transition_entropy',
                            'morning_nap_min', 'afternoon_nap_min', 'evening_nap_min',
                            'late_night_nap_min', 'wake_percent',
                            'light_percent', 'deep_percent', 'rem_percent', 
                            'sleep_onset_adjusted'], # time series features 
                 static_features = ['age', 'sex_male', 'race_white', 'BMI'],
                 date_col = 'sleep_date',
                 label_col = 'label'):
        
        super().__init__()
        # Drop the na values
        df = df.dropna(subset = features + static_features)
        # use person_id as index
        pid_map = {pid: ind for ind, pid in enumerate(df[id_col].unique())}
        df = df.set_index(df[id_col].map(pid_map))
        self.df = df
        self.max_len = max_len
        self.date_col = date_col
        self.label_col = label_col
        if features is None:
            subcohort_cols = [x for x in df.columns if x.startswith('subcohort_')]
            features = [x for x in df.columns if x not in ['person_id', 
                                                            'interval_id', 
                                                            'streak_start', 
                                                            'streak_end',
                                                            'length',
                                                            'drug_exposure_start_datetime',
                                                            'osa_device_date',
                                                            'streak_after',
                                                            'n_diag',
                                                            'label'] + subcohort_cols]
        self.features = features
        self.static_features = static_features
        self.pid_map = pid_map

    def __len__(self):
        return len(self.pid_map)


    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.item()   # convert tensor(33) -> 33 (int)
        df = self.df
        max_len = self.max_len
        features = self.features
        static_features = self.static_features
        date_col = self.date_col
        label_col = self.label_col
        # Get the data for one person, truncate at max length
        df_ind = df.loc[index].sort_values(date_col, ascending=True).iloc[:max_len]

        # return in tensor instead of dataframe
        data = torch.tensor(df_ind[features].values.astype(float))
        # Get static features but don't expand it to the sequence dimension
        data_static = df_ind[static_features].values.astype(float)[0]
        # label positive class
        if type(label_col) == str: # if label is a single column
            label = torch.tensor([float(int(df_ind[label_col].iloc[0]))])
        # I know I am supposed to do more rigorous type checking 
        # # but ain't no one else gonna read my code let alone run it
        else: 
            label = torch.tensor(df_ind[label_col].iloc[0].tolist(), dtype=float)
        # NOTE the below line is for multi-class
        # label = torch.tensor([1 - df_ind['label'].iloc[0], df_ind['label'].iloc[0]], dtype = float)
        
        return data, data_static, label


def get_train_val_test_dataloader(dataset, val_size, test_size, batch_size, g):
    # Get dataset from file list
    total_size = len(dataset)
    indices = torch.randperm(total_size, generator = g)
    val_len = int(val_size*total_size)
    test_len = int(test_size*total_size)

    test_indices = indices[:test_len]
    val_indices = indices[test_len:test_len+val_len]
    train_indices = indices[test_len+val_len:]
    # Create train and test dataset from the indices
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    assert len(train_dataset) + len(val_dataset) + len(test_dataset) == total_size


    # Pass dataset into a dataloader, no need to shuffle because of randperm
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    return train_dataloader, val_dataloader, test_dataloader


###### Function to compute the mean and standard deviation across all samples (and time steps) for input normalisation
def compute_mean_and_std_batch(train_dataloader):
    # Compute the mean and standard deviation from train dataset
    sum_feats = 0.0
    sum_sq_feats = 0.0
    n_total = 0

    for data, _ , _ in train_dataloader:   # time series data: [B, T, F]
        B, T, F = data.shape
        data = data.float()
        sum_feats += data.sum(dim=(0,1))          # sum over batch & time
        sum_sq_feats += (data**2).sum(dim=(0,1))  # sum of squares
        n_total += B * T

    mean = sum_feats / n_total
    var = (sum_sq_feats / n_total) - (mean ** 2)
    std = torch.sqrt(var + 1e-6)

    return mean, std


def compute_mean_and_std_static_batch(train_dataloader):
    # Compute the mean and standard deviation from train dataset
    sum_feats = 0.0
    sum_sq_feats = 0.0
    n_total = 0

    for _, data_static, _ in train_dataloader:   # static data:[B, F]
        B, F = data_static.shape
        data_static = data_static.float()
        sum_feats += data_static.sum(dim=(0))          # sum over batch
        sum_sq_feats += (data_static**2).sum(dim=(0))  # sum of squares
        n_total += B

    mean = sum_feats / n_total
    var = (sum_sq_feats / n_total) - (mean ** 2)
    std = torch.sqrt(var + 1e-6)

    return mean, std

def normalise_features(data, data_static, mean_features, std_features, mean_static, std_static):
    data = torch.tensor(data, dtype=torch.float32)
    data = (data - mean_features[None, None, :]) / std_features[None, None, :]

    data_static = torch.tensor(data_static, dtype=torch.float32)
    data_static = (data_static - mean_static[None, :]) / std_static[None, :]

    return data, data_static


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model=32, n_layers=3, n_heads=4, d_ff=128, max_len=21, dropout_p = 0):
        super().__init__()
        self.encoder = TransformerEncoder(input_dim, d_model, n_layers, n_heads, d_ff, max_len)

        self.dropout = nn.Dropout(p = dropout_p)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.encoder(x)          # (batch, seq_len, d_model)
        x = self.dropout(x)
        # Normalisation and pooling
        x = F.normalize(x, dim = -1) # Normalisation across embedding dimension NOTE bad if accidentally normalising across sequence length
        out = x.mean(dim=1)            # pool over sequence, NOTE might squash the signal if sequence is short?
        # out = self.clf_head(x)           # (batch, 1)
        return out


class ClassifierHead(nn.Module):
    def __init__(self, d_model, out_dim = 1):
        super().__init__()
        self.clf_head = nn.Sequential(
                nn.Linear(d_model, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, out_dim)
                )
    
    def forward(self, x):
        out = self.clf_head(x)
        return out


def get_pred(model, clf, test_dataloader,
             mean_features,
             std_features,
             mean_static,
             std_static,
             device,
             with_static = False):
    
    model = model.to('cuda')
    clf = clf.to('cuda')
    model.eval()
    clf.eval()
    all_preds = []
    all_probas = []
    all_labels = []

    with torch.no_grad():
        for data, data_static, label in test_dataloader:
            # Cast the data to float32 to ensure consistency with model parameters
            data, data_static = normalise_features(data, data_static, mean_features, std_features, mean_static, std_static)
            data = data.to(device)
            data_static = data_static.to(device)
            label = label.to(device)

            # Forward pass
            out = model(data)
            if with_static == True:
                out = torch.concat((out, data_static), dim = 1)
            out = clf(out)
            
            # Predictions
            test_proba = F.sigmoid(out)
            test_pred = test_proba >= 0.5 # cutoff can be modified

            # Move back to cpu and collect
            all_preds.append(test_pred.cpu())
            all_probas.append(test_proba.cpu())
            all_labels.append(label.cpu())

    # Concatenate all results
    test_pred = torch.cat(all_preds, dim=0)
    test_proba = torch.cat(all_probas, dim=0)
    test_labels = torch.cat(all_labels, dim=0)

    # Move back to cpu to assess classifier performance
    test_pred = test_pred.to('cpu')
    test_proba = test_proba.to('cpu')
    test_labels = test_labels.to('cpu')

    return test_pred, test_proba, test_labels



def assess_clf_perf(y_pred, y_prob, y_true, 
                    save_folder,
                    labels = ['Negative (0)', 'Positive (1)']):
    """
    Function to assess classifier performance for sklearn
    """
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    # Classification Report
    print("\nClassification Report:")

    ############### Write performance to file ##################
    with open(f"{save_folder}/cls_metrics.txt", mode = 'w') as f:

        # Metrics
        f.write(f"\nAccuracy: {accuracy_score(y_true, y_pred)}")
        f.write(f"\nPrecision: {precision_score(y_true, y_pred)}")
        f.write(f"\nRecall: {recall_score(y_true, y_pred)}")
        f.write(f"\nF1 Score: {f1_score(y_true, y_pred)}")
        f.write(f"\nROC AUC Score: {roc_auc_score(y_true, y_prob)}")

        # Confusion matrix
        f.write("\nConfusion Matrix:")
        f.write(np.array2string(cm))
        f.write('\n')

        # Confusion matrix
        f.write(classification_report(y_true, y_pred))
        f.write('\n')
        f.write('Feature importance\n')

    ############ Plotting ###############
    # Plot using gridspec for better spacing

    fig = plt.figure(figsize=(15, 7))
    gs = fig.add_gridspec(3, 1, 
                        left=0.1, bottom=0.1, right=0.9, top=0.9,
                        height_ratios=[1.5, 2, 2], hspace=0.5, wspace=0.3)

    # Confusion matrix heatmap
    ax0 = fig.add_subplot(gs[0, 0])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax0)
    ax0.set_title('Confusion Matrix')
    ax0.set_xlabel('Predicted Label')
    ax0.set_ylabel('True Label')

    # ROC curve
    ax1 = fig.add_subplot(gs[1, 0])
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    ax1.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_true, y_prob):.2f}', color='darkorange')
    ax1.plot([0, 1], [0, 1], 'k--', lw=1)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend(loc = "lower right")

    ax2 = fig.add_subplot(gs[2, 0])
    # Probability distribution stratified by true label
    probs_pos_class = y_prob[y_true==1]
    probs_neg_class = y_prob[y_true==0]
    ax2.hist(probs_pos_class, bins=30, alpha=0.6, label='Positive (True=1)', color='blue', density=True)
    ax2.hist(probs_neg_class, bins=30, alpha=0.6, label='Negative (True=0)', color='red', density=True)
    ax2.legend(loc = "lower right")

    # Reset figure size because shap beeswarm messes it up
    fig.set_figheight(12)
    fig.set_figwidth(8)

    # Save figure
    fig.savefig(f"{save_folder}/cls_plot.png", dpi = 300)

    return