import pandas as pd
import numpy as np
import os
import multiprocessing
import yaml
import pickle
from imblearn.over_sampling import RandomOverSampler
from lib.descriptives import compare_two_cohorts_side_by_side

# Plotting
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns

# Modelling
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

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

# Explainability
import shap

# Filter warnings
import warnings
warnings.filterwarnings("ignore")

import matplotlib as mpl

# Set global font sizes
fs_title = 11
fs_normal = 9
fs_tick = 8

mpl.rcParams['font.size'] = fs_normal            # Base font size
mpl.rcParams['axes.titlesize'] = fs_title      # Axes title font size
mpl.rcParams['axes.labelsize'] = fs_normal       # Axis label font size
mpl.rcParams['xtick.labelsize'] = fs_tick      # X tick label size
mpl.rcParams['ytick.labelsize'] = fs_tick      # Y tick label size
mpl.rcParams['legend.fontsize'] = fs_tick      # Legend font size
mpl.rcParams['figure.titlesize'] = fs_title    # Figure title size


def assess_clf_perf(clf, X_test, y_true, 
                    save_folder,
                    features,
                    feature_rename_map = None,
                    labels = ['Negative (0)', 'Positive (1)']):
    """
    Function to assess classifier performance for sklearn
    """

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:,1]
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    # Classification Report
    print("\nClassification Report:")

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

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
    
    # Native feature importance
    feat_imp = pd.Series(clf.get_booster().get_score(importance_type='gain')).sort_values(ascending=False)
    # feat_imp = feat_imp.rename(feature_rename_map)
    feat_imp = pd.DataFrame(feat_imp)

    feat_imp.to_csv(f"{save_folder}/cls_metrics.txt", mode = 'a')

    ############ Plotting ###############
    # Plot using gridspec for better spacing

    fig = plt.figure(figsize=(18, 25))
    gs = fig.add_gridspec(4, 2, 
                        left=0.1, bottom=0.1, right=0.9, top=0.9,
                        height_ratios=[1.5, 2, 2, 1.5], hspace=0.5, wspace=0.3)

    # Confusion matrix heatmap
    ax0 = fig.add_subplot(gs[0, 0])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax0)
    ax0.set_title('Confusion Matrix')
    ax0.set_xlabel('Predicted Label')
    ax0.set_ylabel('True Label')

    # ROC curve
    ax1 = fig.add_subplot(gs[0, 1])
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    ax1.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_true, y_prob):.2f}', color='darkorange')
    ax1.plot([0, 1], [0, 1], 'k--', lw=1)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend(loc="lower right")

    # Native XGBoost gain importance
    ax2 = fig.add_subplot(gs[1, :])
    xgb.plot_importance(clf, importance_type='gain', show_values=False, ax = ax2,
                        max_num_features = 15)
    # Map the labels
    original_labels = [tick.get_text() for tick in ax2.get_yticklabels()]
    if feature_rename_map is not None:
        ax2.set_yticklabels([feature_rename_map.get(label, label) for label in original_labels])
    # aesthetic adjustment
    ax2.grid(axis='y', visible=False)
    # Change bar color
    for bar in ax2.patches:
        bar.set_color('#2b60a0')
    ax2.set_title('XGBoost Feature Importance (Gain)')

    # SHAP summary plot
    ax3 = fig.add_subplot(gs[2, :])
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, feature_names=features, max_display=15)
    plt.sca(ax3)
    # Map the labels
    original_labels = [tick.get_text() for tick in ax3.get_yticklabels()]
    if feature_rename_map is not None:
        ax3.set_yticklabels([feature_rename_map.get(label, label) for label in original_labels])
    ax3.set_title("SHAP Feature Importance")

    
    # Adjust font size manually
    ax3.title.set_fontsize(fs_title)
    ax3.xaxis.label.set_fontsize(fs_normal)
    ax3.yaxis.label.set_fontsize(fs_normal)
    ax3.tick_params(axis='both', labelsize=fs_tick)

    # Offset gain and SHAP plots to the right
    pos = ax2.get_position()
    ax2.set_position([pos.x0 + 0.2, pos.y0, pos.width * 0.8, pos.height*0.9])

    pos = ax3.get_position()
    ax3.set_position([pos.x0 + 0.2, pos.y0, pos.width * 0.9, pos.height*0.9])

    ######## fix beeswarm colour bar ############
    cb_ax = [x for x in fig.axes if x not in [ax0, ax1, ax2, ax3]]
    # first colour bar is for confusion matrix, ignore that one
    cb_ax0, cb_ax1 = cb_ax[0], cb_ax[1]

    # Adjust font size
    cb_ax1.yaxis.label.set_fontsize(fs_normal)

    # Adjust positions
    pos = cb_ax1.get_position()
    cb_ax1.set_position([pos.x0 + 0.1, pos.y0, pos.width, pos.height*0.9])

    # Size of tick params
    cb_ax1.tick_params(labelsize=fs_tick, length=0)

    ax4 = fig.add_subplot(gs[3, 0])
    # Probability distribution stratified by true label
    probs_pos_class = y_prob[y_true==1]
    probs_neg_class = y_prob[y_true==0]
    ax4.hist(probs_pos_class, bins=30, alpha=0.6, label='Positive (True=1)', color='blue', density=True)
    ax4.hist(probs_neg_class, bins=30, alpha=0.6, label='Negative (True=0)', color='red', density=True)
    ax4.legend(loc = "lower right")
    ax4.set_title("Predicted probability by true class label")

    # Reset figure size because shap beeswarm messes it up
    fig.set_figheight(16)
    fig.set_figwidth(8)

    # Save figure
    fig.savefig(f"{save_folder}/cls_plot.png", dpi = 300)

    return fig



def add_mean(feat_list):
    """
    If coefficient of variation is selected as one of the features,
    Add the mean as well"""
    cv_list = [x for x in feat_list if x.endswith('_cv')]
    mean_list = [x.replace('_cv', '_mean') for x  in cv_list]
    # Add the mean features to feature list
    feat_list.extend(mean_list)
    # remove duplicates
    feat_list = list(set(feat_list))

    return feat_list

def prepare_data(sleep_data_collection_date, # Dataframe
                person_df,
                person_features,
                physical_measurement_df,
                physical_measurement_features,
                comp,
                sleep_agg_stats_l, # one person should only has one row
                s1,
                s2,
                cohens_d_cutoff,
                incl_mean_if_cv_selected,
                categorical_vars,
                save_path = None):
    
    ######### Sleep features #########
    df1 = sleep_agg_stats_l.loc[sleep_agg_stats_l['subcohort'] == s1]
    df1['label'] = 1
    df2 = sleep_agg_stats_l.loc[sleep_agg_stats_l['subcohort'] == s2]
    df2['label'] = 0
    df = pd.concat([df1, df2], axis=0).reset_index(drop=True)

    # List of sleep features to use - filter by effect size
    if cohens_d_cutoff:
        sleep_features = comp.loc[np.abs(comp['cohens_d']) >= cohens_d_cutoff, 'feature'].tolist()
    else:
        sleep_features = comp['feature'].tolist()

    if incl_mean_if_cv_selected:
        sleep_features = add_mean(sleep_features)
    df = df[['person_id'] + sleep_features + ['label']]

    ######### Demographic features #########
    # Compute age using sleep data collection date as a reference
    person_df = person_df.merge(sleep_data_collection_date)
    person_df['age'] = ((pd.to_datetime(person_df['streak_start']) - \
                         pd.to_datetime(person_df['date_of_birth'].str[0:10])).dt.days/365).astype(int)
    person_df = person_df[['person_id'] + person_features]
    
    ######### Physical measurement features #########
    # Select one closest to sleep collection date
    physical_measurement_df = physical_measurement_df.merge(sleep_data_collection_date)
    physical_measurement_df['diff'] = (pd.to_datetime(physical_measurement_df['streak_start']) - \
                                        pd.to_datetime(physical_measurement_df['measurement_date'])).dt.days
    physical_measurement_df = physical_measurement_df.merge(
                                physical_measurement_df[['person_id', 'diff']].groupby('person_id', as_index=False).min(),
                                on = ['person_id', 'diff'])
    physical_measurement_df = physical_measurement_df[['person_id'] + physical_measurement_features]

    # Join everything together
    df = df.merge(person_df).merge(physical_measurement_df, how = 'left')

    # Set categorical variable
    df[categorical_vars] = df[categorical_vars].astype('category')

    if save_path:
        df.to_csv(save_path, index=False)

    return df

    
def create_mdl(df, features, label_col = 'label', imb_data = False):

    # Create X and y
    X = df[features]
    y = df[label_col]
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        stratify = y, random_state=41)
    
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



def main():

    with open('conf/mdl_sleep.yaml') as f:
        config = yaml.safe_load(f)
    
    # Extract common configs
    categorical_vars = config['categorical_vars']
    person_df = pd.read_csv(config['person_df'])
    physical_measurement_df = pd.read_csv(config['physical_measurement_df'])
    streaks = pd.read_csv(config['sleep_streaks'])
    sleep_agg_stats = pd.read_csv(config['sleep_agg_stats'])
    person_features = config['person_features']
    physical_measurement_features = config['physical_measurement_features']
    exclude_std = config['exclude_std']
    features_excl = config['features_excl']
    incl_mean_if_cv_selected = config['incl_mean_if_cv_selected']
    save_folder = config['save_folder']
    feature_rename_map = config['feature_rename_map']

    # Extract model-specific config
    model_config = config['models']
    for mdl_name, conf in model_config.items():
        if not conf['run']:
            continue
        print(f"Creating {mdl_name}...")
        ############## Model specific config #####################
        s1 = conf['subcohorts'][0]
        s2 = conf['subcohorts'][1]
        cohens_d_cutoff = conf['cohens_d_cutoff']
        cohort = pd.read_csv(conf['cohort_df'])
        imb_data = conf['imb_data'] if 'imb_data' in conf.keys() else False
        # Create model-specific save folder
        save_folder_mdl = f"{save_folder}/{mdl_name}/"
        save_folder_mdl_base = f"{save_folder}/{mdl_name}_base/"
        save_folder_mdl_sleep_only = f"{save_folder}/{mdl_name}_sleep_only/"
        os.makedirs(save_folder_mdl, exist_ok=True)
        os.makedirs(save_folder_mdl_base, exist_ok=True)
        os.makedirs(save_folder_mdl_sleep_only, exist_ok=True)

        ############### restriction on sleep data collection date ##################
        sleep_date_res = conf['sleep_date_must_be_before']
        
        if sleep_date_res:
            for _, res in sleep_date_res.items():
                date_col = res['date_col']
                sleep_date_res_df = pd.read_csv(res['fpath'])
                sleep_date_res_df[date_col] = pd.to_datetime(sleep_date_res_df[date_col].str[0:10])
                sleep_date_res_df = sleep_date_res_df[['person_id', date_col]].groupby('person_id', as_index=False).min()
                streaks = streaks.merge(sleep_date_res_df, how = 'left')
                streaks = streaks.loc[(streaks['streak_end']>=streaks[date_col]) | streaks[res['date_col']].isna()]
        
        # print(streaks.head(5))
        
        # Check that each row of streaks df have unique interval id
        if streaks.shape[0] != streaks['interval_id'].nunique():
            raise Exception('Rows of streaks df do not have unique interval ID')

        ############## NOTE might be changed: take the longest streak ##############
        longest_streak_by_person = sleep_agg_stats.merge(streaks, 
                                    on = ['person_id', 'interval_id']).set_index('interval_id')\
                                    .groupby('person_id').idxmax()['length']
        sleep_agg_stats_l = sleep_agg_stats.loc[sleep_agg_stats['interval_id'].isin(longest_streak_by_person)].copy()
        # Set sleep data collection date as the longest streak by person
        sleep_data_collection_date = streaks.loc[streaks['interval_id'].isin(longest_streak_by_person), 
                                                 ['person_id', 'streak_start']]

        # print(streaks.loc[streaks['interval_id'].isin(longest_streak_by_person)].sort_values('person_id'))

        # Check that each row of streaks df have unique interval id
        if sleep_agg_stats_l.shape[0] != sleep_agg_stats_l['interval_id'].nunique():
            raise Exception('Rows of sleep_agg_stats_l df do not have unique interval ID (Check 2)')
        
        ############# Descriptive statistics for comparison ###############
        sleep_agg_stats_l = sleep_agg_stats_l.merge(cohort, on = 'person_id')

        # Check that each row of cohort df have unique interval id
        if cohort.shape[0] != cohort['person_id'].nunique():
            raise Exception('Rows of cohort df do not have unique interval ID')
        
        # Check that each row of streaks df have unique interval id
        if sleep_agg_stats_l.shape[0] != sleep_agg_stats_l['interval_id'].nunique():
            raise Exception('Rows of sleep_agg_stats_l df do not have unique interval ID (Check 2)')
        
        sleep_features = [x for x in sleep_agg_stats.columns if \
                        (x not in ['person_id', 'interval_id', 'subcohort'])]
        if exclude_std:
            sleep_features = [x for x in sleep_features if not x.endswith('_std')]
        if features_excl:
            sleep_features = [x for x in sleep_features if x not in features_excl]
        # Descriptives stats 
        comp = compare_two_cohorts_side_by_side(sleep_agg_stats_l.loc[sleep_agg_stats_l['subcohort'] == s1],
                                            sleep_agg_stats_l.loc[sleep_agg_stats_l['subcohort'] == s2],
                                            sleep_features,
                                            label1=s1, label2=s2,
                                            save_folder = save_folder_mdl)
    
        ########### Prepare the data for modelling ########### 
        df = prepare_data(sleep_data_collection_date, # Dataframe
                        person_df,
                        person_features,
                        physical_measurement_df,
                        physical_measurement_features,
                        comp,
                        sleep_agg_stats_l, # one person should only has one row
                        s1,
                        s2,
                        cohens_d_cutoff,
                        incl_mean_if_cv_selected,
                        categorical_vars,
                        # save_path = None) # Let's not save the dataframe
                        save_path = f"{save_folder_mdl}/mdl_df.csv")

        # Assess the performance of the classifier
        clf, X_test, y_test = create_mdl(df, [x for x in df.columns if x not in ['person_id', 'label']], imb_data)

        ##################### Save model and performance metrics ####################
        # Save model
        with open(f"{save_folder_mdl}/model.pkl",'wb') as f:
            pickle.dump(clf, f)

        assess_clf_perf(clf, X_test, y_test,
                        save_folder_mdl,
                        [x for x in df.columns if x not in ['person_id', 'label']],
                        feature_rename_map,
                        labels = [s2, s1]) # Note that the label is the other way round because s2 is the 'negative' one
    
        ################# Fit base model ###################
        feat_excl_base = ['person_id', 'label'] + sleep_features
        clf_base, X_test_base, y_test_base = create_mdl(df, 
                                            [x for x in df.columns if x not in feat_excl_base])
        
        assess_clf_perf(clf_base, X_test_base, y_test_base,
                        save_folder_mdl_base,
                        [x for x in df.columns if x not in feat_excl_base],
                        feature_rename_map,
                        labels = [s2, s1]) # Note that the label is the other way round because s2 is the 'negative' one

        ################# Fit model with sleep features only ###################
        clf_sleep, X_test_sleep, y_test_sleep = create_mdl(df, [x for x in df.columns if x in sleep_features])
        
        assess_clf_perf(clf_sleep, X_test_sleep, y_test_sleep,
                        save_folder_mdl_sleep_only,
                        [x for x in df.columns if x in sleep_features],
                        feature_rename_map,
                        labels = [s2, s1]) # Note that the label is the other way round because s2 is the 'negative' one


    return


if __name__ == '__main__':
    main()