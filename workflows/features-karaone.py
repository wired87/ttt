"""
features-karaone.py
Extraction of features from KaraOne dataset.
"""
import os
import sys
from rich.console import Console

sys.path.append(os.getcwd())
from utils.config import load_config, line_separator
from utils.karaone import KaraOneDataLoader


if __name__ == "__main__":
    console = Console()
    args = load_config(key="karaone")

    karaone = KaraOneDataLoader(
        data_dir=args["data_dir"],
        subjects="all",
        sampling_freq=1000,
        num_milliseconds_per_trial=4900,
        verbose=True,
        console=console,
    )

    karaone.process_data(
        epoch_type="thinking",
        pick_channels=[-1],
        num_neighbors=5,
        verbose=False,
    )
    karaone.epochs_info(verbose=True)
    labels = karaone.all_epoch_labels

    karaone.extract_features(
        features_dir=args["features_dir"],
        epoch_type="thinking",
    )

    features = karaone.load_features(epoch_type="thinking", verbose=True)

    flattened_features, flattened_labels = karaone.flatten(
        features, labels, verbose=True
    )