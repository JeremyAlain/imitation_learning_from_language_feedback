import string
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import sem

alphabet = string.ascii_uppercase


def calculate_win_rate_between_all_method_rankings(
    data: pd.DataFrame, method_names: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results_per_method_name: Dict[str, List[int]] = {}
    for method_name in method_names:
        assert method_name in data.columns
        results_per_method_name["{}_ranking".format(method_name)] = []
    number_of_methods = len(method_names)

    ranking_column_name = "Rank the summaries according to quality: Summary A: {{Summary_A}} Summary B:  {{Summary_B}}  Summary C: {{Summary_C}}"
    for i in range(data.shape[0]):
        current_row = data.iloc[i]
        for summary_index in range(number_of_methods):
            assert current_row[ranking_column_name][summary_index][
                "text"
            ] == "Summary {}".format(alphabet[summary_index])

        current_row_ranking = current_row[ranking_column_name]
        for method_name in method_names:
            current_method_placement_index = current_row[
                "placement_indices_{}".format(method_name)
            ]
            current_method_ranking = current_row_ranking[
                current_method_placement_index
            ]["rank"]
            results_per_method_name["{}_ranking".format(method_name)].append(
                current_method_ranking
            )

    data = pd.concat([data, pd.DataFrame(results_per_method_name)], axis=1)
    method_ranking_names = [
        "{}_ranking".format(method_name) for method_name in method_names
    ]
    # convert the ranking to standard competition ranking i.e. 1,2,2,4 instead of 1,3,3,
    data[method_ranking_names] = (
        data[method_ranking_names].rank(axis=1, method="average").astype(int)
    )

    win_rate_results = {}
    standard_error_results = {}
    ties_results = {}
    for method_name in method_names:
        for other_method_name in method_names:
            if method_name == other_method_name:
                continue
            else:
                wins = (
                    data["{}_ranking".format(method_name)]
                    < data["{}_ranking".format(other_method_name)]
                )
                ties = (
                    data["{}_ranking".format(method_name)]
                    == data["{}_ranking".format(other_method_name)]
                )
                assert np.mean(wins & ties) == 0
                assert len(wins) == len(ties)
                win_rate_results[
                    "{}_vs_{}_win_rate_mean".format(method_name, other_method_name)
                ] = [np.mean(wins + 0.5 * ties)]
                standard_error_results[
                    "{}_vs_{}_win_rate_standard_error".format(
                        method_name, other_method_name
                    )
                ] = [sem(wins + 0.5 * ties)]
                ties_results[
                    "{}_vs_{}_ties_mean".format(method_name, other_method_name)
                ] = [np.mean(ties)]

    return (
        data,
        pd.DataFrame(win_rate_results),
        pd.DataFrame(standard_error_results),
        pd.DataFrame(ties_results),
    )
