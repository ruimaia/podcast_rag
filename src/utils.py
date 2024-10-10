import json
import numpy as np
import pandas as pd
import os

def str2bool(string):
    string = string.lower()
    str2val = {"true": True, "false": False}
    if string in str2val:
        return str2val[string]
    else:
        raise ValueError(f"Expected one of {set(str2val.keys())}, got {string}")

def save_output_file(data, output_dir, stage, filename, overwrite):
    if os.path.exists(output_dir) and not overwrite:
        raise ValueError("Specified `output_dir` already exists. Either change `output_dir` or set `overwrite` to True.")

    output_dir = os.path.join(output_dir, stage)
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, filename)

    if type(data) == dict:
        assert filename.endswith(".json")
        with open(output_path, "w") as fp:
            json.dump(data, fp, ensure_ascii=False, indent=2)

    if type(data) == np.ndarray:
        assert filename.endswith(".npy")
        np.save(output_path, data, allow_pickle=True)

    if type(data) == pd.DataFrame:
        assert filename.endswith(".csv")
        data.to_csv(output_path, index=False)
