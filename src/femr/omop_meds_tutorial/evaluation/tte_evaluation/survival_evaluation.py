from pycox.models import DeepHitSingle
from pycox.evaluation import EvalSurv
import torchtuples as tt

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import json
import os

np.random.seed(42)
torch.manual_seed(42)


# base = '/data/processed_datasets/processed_datasets/mimic/phenotype/'
# phenotype = 'masld'
# model = 'motor'

# path = base + phenotype + '/' + model + '/features_with_label/'

def tte_evaluation(train_set,test_set,task,model_name):
    print(f"task is {task}, model is {model_name} initial length {len(train_set)},{len(test_set)}")
    train_set = train_set[train_set["time_to_event"]<=330]
    test_set = test_set[test_set["time_to_event"]<=330]
    print(f"after length {len(train_set)},{len(test_set)}")
    val = train_set.sample(frac = 0.1, replace = False)
    train_set = train_set.drop(val.index)
    # train[~train.boolean_value].time_to_event_days.plot.hist(bins = 1000)

    # Transform labels
    num_durations = 100
    labtrans = DeepHitSingle.label_transform(num_durations)
    get_target = lambda df: (df['time_to_event'].values, df['tte_label'].values) 

    # Extract features and labels
    x_train = np.stack(train_set.features.values)
    y_train = labtrans.fit_transform(*get_target(train_set))
    train = (x_train, y_train)

    x_val = np.stack(val.features.values)
    y_val = labtrans.transform(*get_target(val))
    val = (x_val, y_val)

    x_test = np.stack(test_set.features.values)
    durations_test, events_test = get_target(test_set)

    # Define one layer NN for Deephit
    in_features = x_train.shape[1]
    out_features = labtrans.out_features

    net = tt.practical.MLPVanilla(in_features, [], out_features)

    # Train Deephit
    model = DeepHitSingle(net, tt.optim.Adam, duration_index=labtrans.cuts)
    epochs, batch = 100, 2048
    log = model.fit(x_train, y_train, batch, epochs, [tt.callbacks.EarlyStopping()], val_data=val)

    # Predict on test set
    surv = model.predict_surv_df(x_test)

    # Compute boostrapped performance
    results = {'C-Index':[], 'Int-Brier':[]}
    time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
    for boot in tqdm(range(100)):
        sample = np.random.choice(len(x_test), len(x_test))
        ev = EvalSurv(surv[sample], durations_test[sample], events_test[sample], censor_surv='km')
        results['C-Index'].append(ev.concordance_td('antolini'))
        results['Int-Brier'].append(ev.integrated_brier_score(time_grid))

    # Display
    for metric in results:
        print('{}: {:.2f} ({:.2f})'.format(metric, np.mean(results[metric]), np.std(results[metric])))
    return results


old_phenotype_dir = Path("/shared/share_mala/zj2398/mimic/task/old_phenotype")
# old_phenotype_path
for task in ["celiac","masld","stroke"]:
    old_phenotype_path = old_phenotype_dir/"results"/task
    for subdir in old_phenotype_path.iterdir():
        train_path = subdir/"features_with_label"/"train.parquet"
        test_path = subdir/"features_with_label"/"test.parquet"
        train_parquet = pd.read_parquet(train_path)
        test_parquet = pd.read_parquet(test_path)
        # print(train_parquet)
        # print(test_parquet)
        tte_results = tte_evaluation(train_parquet,test_parquet,task,subdir.name)
        metrics = []
        metrics['C-Index'] = (np.mean(tte_results['C-Index']), np.std(tte_results['C-Index']))
        metrics['Int-Brier'] = (np.mean(tte_results['Int-Brier']), np.std(tte_results['Int-Brier']))
        save_path = old_phenotype_dir/"results_1st_year"/task/subdir.name/"tte.json"
        os.makedirs(save_path,exist_ok=True)
        with open(save_path,"w") as f:
            json.dump(metrics,f)
