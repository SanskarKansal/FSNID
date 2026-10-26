import numpy as np
import os
import torch
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from evaluate_features import FeatureClassifier
from feature_selection_methods.fsnid_fs import fsnid_selection
from feature_selection_methods.brown_fs import brown_selection
from feature_selection_methods.firefly_fs import firefly_selection
from feature_selection_methods.lasso_fs import lasso_selection
from feature_selection_methods.pi_fs import pi_selection
import time
import json
from generate_synthetic_data import generate_synthetic_nid_dataset


def parse_arguments():
    parser = argparse.ArgumentParser(description='Feature Selection and Classification for Network Intrusion Detection')
    parser.add_argument('--nme', type=str, default='SYNTHETIC',
                        help='Name of the dataset (default: SYNTHETIC)')
    parser.add_argument('--selection_method', type=str, default='fsnid',
                        choices=['fsnid', 'brown', 'firefly', 'lasso', 'pi'],
                        help='Feature selection method to use (default: fsnid)')
    parser.add_argument('--model_type', type=str, default='MLP',
                        choices=['MLP', 'LSTM', 'TCN', 'GRU'],
                        help='Type of model to evaluate features with (default: MLP)')
    parser.add_argument('--num_iterations', type=int, default=10,
                        help='Number of iterations for MINE / feature selection (default: 10)')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Training epochs for classification evaluation (default: 15)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for classifier (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for classifier training (default: 64)')
    args = parser.parse_args()
    return args

def get_selection_class(method):
    selection_mapping = {
        'fsnid': fsnid_selection,
        'brown': brown_selection,
        'firefly': firefly_selection,
        'lasso': lasso_selection,
        'pi': pi_selection
    }
    if method not in selection_mapping:
        raise ValueError(f"Invalid selection method '{method}'. Choose from {list(selection_mapping.keys())}.")
    return selection_mapping[method]

def save_metrics(metrics, data_dir, nme, selection_method):
    metrics_filename = f'{selection_method}_{nme}_metrics.json'
    metrics_path = os.path.join(data_dir, metrics_filename)
    try:
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {metrics_path}")
    except Exception as e:
        print(f"Failed to save metrics to JSON: {e}")

def main():
    args = parse_arguments()
    nme = args.nme
    selection_method = args.selection_method
    model_type = args.model_type
    num_iterations = args.num_iterations
    epochs = args.epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size

    print("=" * 60)
    print(" FSNID: Feature Selection for Network Intrusion Detection")
    print("=" * 60)
    print(f"Dataset Name (nme): {nme}")
    print(f"Feature Selection Method: {selection_method}")
    print(f"Model Type: {model_type}")
    print(f"Feature Selection Iterations: {num_iterations}")
    print(f"Classifier Epochs: {epochs}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Batch Size: {batch_size}")
    print("-" * 60)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    X_path = os.path.join(data_dir, f'{nme}_X.npy')
    Y_path = os.path.join(data_dir, f'{nme}_Y.npy')

    if (not os.path.exists(X_path) or not os.path.exists(Y_path)) and nme == 'SYNTHETIC':
        print(f"Dataset {nme} not found in {data_dir}. Generating synthetic dataset (10,000 samples)...")
        generate_synthetic_nid_dataset(output_dir=data_dir, prefix=nme)

    if not os.path.exists(X_path):
        raise FileNotFoundError(f"File not found: {X_path}")
    if not os.path.exists(Y_path):
        raise FileNotFoundError(f"File not found: {Y_path}")

    print("Loading data...")
    X = np.load(X_path)
    y = np.load(Y_path)
    print(f"Original Shapes -> X: {X.shape}, y: {y.shape}")

    df = pd.DataFrame(X)
    df_min = df.min()
    df_max = df.max()
    denom = df_max - df_min
    denom[denom == 0] = 1.0
    df = (df - df_min) / denom
    df = df.fillna(0)
    X_preprocessed = df.to_numpy()
    y_preprocessed = y.flatten().astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X_preprocessed, y_preprocessed, test_size=0.2, random_state=42, stratify=y_preprocessed
    )
    print(f"Data Split -> Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    selection_class = get_selection_class(selection_method)
    print(f"Initializing feature selection using '{selection_method}' method...")
    if selection_method == 'fsnid':
        feature_selector = selection_class(X_train, y_train, num_iterations=num_iterations, model_type=model_type)
    elif selection_method == 'brown':
        feature_selector = selection_class(X_train, y_train, num_iterations=num_iterations)
    else:
        feature_selector = selection_class(X_train, y_train)

    print("Running feature selection...")
    start_time = time.time()
    feats = feature_selector.run_main()
    end_time = time.time()
    selection_duration = end_time - start_time
    print(f"Feature Selection Completed in {selection_duration:.2f} seconds.")
    print(f"Selected Features ({len(feats)}/{X.shape[1]}): {feats}")

    print("-" * 60)
    print(f"Evaluating selected features with {model_type} classifier...")
    classifier = FeatureClassifier(
        X_train, X_test, y_train, y_test, feats,
        model_type=model_type, epochs=epochs, batch_size=batch_size,
        learning_rate=learning_rate, expts=1
    )
    metrics = classifier.run_main()

    final_acc = metrics['accuracy'][0][-1] * 100
    final_f1 = metrics['f1_score'][0][-1]
    final_precision = metrics['precision'][0][-1]
    final_recall = metrics['recall'][0][-1]
    final_fpr = metrics['false_positive_rate'][0][-1]

    print("=" * 60)
    print(" FINAL EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Dataset:                      {nme}")
    print(f"Feature Selection Method:     {selection_method.upper()}")
    print(f"Total Original Features:      {X.shape[1]}")
    print(f"Features Retained:            {len(feats)} ({len(feats)/X.shape[1]*100:.1f}%)")
    print(f"Selected Feature Indices:     {feats}")
    print(f"Selection Time:               {selection_duration:.2f} s")
    print(f"Test Accuracy:                {final_acc:.2f}%")
    print(f"Test F1-Score (Weighted):     {final_f1:.4f}")
    print(f"Test Precision (Weighted):    {final_precision:.4f}")
    print(f"Test Recall (Weighted):       {final_recall:.4f}")
    print(f"Average False Positive Rate:  {final_fpr:.4f}")
    print("=" * 60)

    save_metrics(metrics, data_dir, nme, selection_method)
    np.save(os.path.join(data_dir, f'fsnid_fs_{nme}_{selection_method}.npy'), np.array(feats))
    np.save(os.path.join(data_dir, f'fsnid_{nme}_{selection_method}_time.npy'), np.array([selection_duration]))

if __name__ == '__main__':
    main()
