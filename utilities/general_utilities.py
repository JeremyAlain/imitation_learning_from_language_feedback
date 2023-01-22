import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def split_off_dataset_with_proportional_subreddit_distribution(
    data: pd.DataFrame, target_dataset_size: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = data.reset_index(drop=True)
    initial_data_size = data.shape[0]
    sampling_frac = float(target_dataset_size / initial_data_size)

    assert "subreddit" in data.columns.tolist()
    # sample proportionally to the subreddit categories
    data_to_split_off = data.groupby("subreddit", group_keys=False).apply(
        lambda x: x.sample(frac=sampling_frac, replace=False)
    )
    data = data.drop(data_to_split_off.index, axis=0)
    data = data.reset_index(drop=True)

    # if sampling with sampling frac lead to too many or to few samples than target_dataset_size indicates, add or remove some
    while data_to_split_off.shape[0] != target_dataset_size:
        if data_to_split_off.shape[0] < target_dataset_size:
            sample_difference = target_dataset_size - data_to_split_off.shape[0]
            random_indices = np.random.choice(
                data.shape[0], size=sample_difference, replace=False
            )
            random_data = data.iloc[random_indices]
            full_data_to_split_off = [data_to_split_off, random_data]
            data_to_split_off = pd.concat(full_data_to_split_off)

            data = data.drop(random_data.index, axis=0)
        else:
            sample_difference = data_to_split_off.shape[0] - target_dataset_size
            data_to_split_off = data_to_split_off.reset_index(drop=True)
            indices_to_remove = np.random.choice(
                data_to_split_off.index.tolist(), size=sample_difference, replace=False
            )
            data_to_remove = data_to_split_off.iloc[indices_to_remove]
            full_dataset = [data, data_to_remove]
            data = pd.concat(full_dataset)
            data_to_split_off = data_to_split_off.drop(indices_to_remove, axis=0)

    assert data_to_split_off.shape[0] == target_dataset_size

    data_to_split_off = data_to_split_off.reset_index(drop=True)
    data.reset_index(drop=True)
    return data, data_to_split_off


def assert_saved_dataframe_is_equal_to_dataframe(
    dataframe_path: str, dataframe: pd.DataFrame
) -> None:
    assert os.path.splitext(dataframe_path)[-1] == ".jsonl"
    if os.path.isfile(dataframe_path):
        saved_dataframe = pd.read_json(dataframe_path, lines=True, orient="records")
        assert (
            saved_dataframe.shape == dataframe.shape
        ), "Size of dataset {} to split off does not match existing split off dataset"
        for i in range(saved_dataframe.shape[0]):
            assert (
                saved_dataframe["id"].iloc[i] == dataframe["id"].iloc[i]
            ), "At loction {}: {} vs {}".format(
                i, saved_dataframe["id"].iloc[i], dataframe["id"].iloc[i]
            )
            assert saved_dataframe.iloc[i].equals(
                dataframe.iloc[i]
            ), "Saved dataset {} does not match provided dataset".format(dataframe_path)
        assert saved_dataframe.equals(dataframe)


def save_list_of_dicts_to_jsonl(
    list_of_dicts: List[Dict[str, Any]], save_path: str
) -> None:
    with open(save_path, "w") as output_file:
        for line in list_of_dicts:
            json.dump(line, output_file)
            output_file.write("\n")


def print_columns_of_dataframe(dataframe: pd.DataFrame, columns: List[str]) -> None:
    data_size = dataframe.shape[0]
    for i in range(data_size):
        print("Sample {} \n".format(i))
        row = dataframe.iloc[i]
        for column in columns:
            print("{}: {} \n".format(column, row[column]))
        print("#################################### \n\n")
