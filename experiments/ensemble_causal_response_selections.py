import os
import string
from typing import Dict, List

import click
import numpy as np
import pandas as pd

alphabet = string.ascii_uppercase


@click.command()
@click.option("--causal_response_prompt_1_results_path", required=True, type=str)
@click.option("--results_folder", required=True, type=str)
@click.option("--output_path", required=True, type=str)
@click.option("--select_top_n", required=True, type=int, default=1)
def main(
    causal_response_prompt_1_results_path: str,
    results_folder: str,
    output_path: str,
    select_top_n: int,
) -> None:
    causal_response_prompt_1_results = pd.read_json(
        causal_response_prompt_1_results_path, lines=True
    )
    original_summary_name = "generated_summary_for_feedback"
    if "generated_summary_C" in causal_response_prompt_1_results.columns:
        # this is for old refinements like dev dataet where naming scheme was different.
        causal_response_prompt_1_results.rename(
            columns={"generated_summary_C": "generated_summary_for_feedback"},
            inplace=True,
        )

    if "generated_test_summary_A" in causal_response_prompt_1_results.columns:
        # in iterative refinement schemes the original summaries that we provide feedback on are called different.y
        original_summary_name = "generated_test_summary_A"

    causal_response_results = []
    for prompt_index in range(1, 6):
        dataframe = pd.read_json(
            os.path.join(
                results_folder,
                "causal_response_prompt_{}_calculations.jsonl".format(prompt_index),
            ),
            lines=True,
            orient="records",
        )
        causal_response_results.append(dataframe)
        if prompt_index > 0:
            assert (
                causal_response_results[0].shape[0]
                == dataframe.shape[0]
                == causal_response_prompt_1_results.shape[0]
            )
    number_of_samples = causal_response_results[0].shape[0]
    ensemble_refinement_selection_log_probabilities = []
    ensemble_ordered_refinement_selection = []
    ensemble_selected_refinements: Dict[str, List[str]] = {}
    for top_n in range(1, select_top_n + 1):
        if top_n == 1:
            ensemble_selected_refinements["ensemble_selected_refinement"] = []
        else:
            ensemble_selected_refinements[
                "ensemble_selected_refinement_top_{}".format(top_n)
            ] = []

    ids = []
    subreddits = []
    titles = []
    posts = []
    feedbacks = []
    original_summaries = []

    for i in range(number_of_samples):
        ids.append(causal_response_prompt_1_results["id"].iloc[i])
        subreddits.append(causal_response_prompt_1_results["subreddit"].iloc[i])
        titles.append(causal_response_prompt_1_results["title"].iloc[i])
        posts.append(causal_response_prompt_1_results["post"].iloc[i])
        feedbacks.append(causal_response_prompt_1_results["feedback"].iloc[i])
        original_summaries.append(
            causal_response_prompt_1_results[original_summary_name].iloc[i]
        )
        all_refinement_selection_log_probabilities = []
        for prompt_index in range(5):
            refinement_selection_log_probabilities = causal_response_results[
                prompt_index
            ].iloc[i]["causal_response_positive_probability"]
            all_refinement_selection_log_probabilities.append(
                refinement_selection_log_probabilities
            )
        average_refinement_log_probabilities = np.array(
            all_refinement_selection_log_probabilities
        ).mean(axis=0)
        ensemble_refinement_selection_log_probabilities.append(
            average_refinement_log_probabilities
        )
        selected_refinement_indices = np.argsort(average_refinement_log_probabilities)[
            ::-1
        ]

        refinement_selection_letters = []
        selected_refinement_indices_top_n = selected_refinement_indices[:select_top_n]
        for top_n_indices in selected_refinement_indices_top_n:
            refinement_selection_letters.append(alphabet[top_n_indices])

        for j, refinement_selection_letter in enumerate(refinement_selection_letters):
            top_n = j + 1
            selected_refinement = causal_response_prompt_1_results.iloc[i][
                "generated_refinement_{}".format(refinement_selection_letter)
            ]
            if top_n == 1:
                ensemble_selected_refinements["ensemble_selected_refinement"].append(
                    selected_refinement
                )
            else:
                ensemble_selected_refinements[
                    "ensemble_selected_refinement_top_{}".format(top_n)
                ].append(selected_refinement)
        ordered_refinement_selection_letters = [
            alphabet[refinement_index]
            for refinement_index in selected_refinement_indices
        ]
        ensemble_ordered_refinement_selection.append(
            ordered_refinement_selection_letters
        )

    dataframe = pd.DataFrame(
        {
            "id": ids,
            "subreddit": subreddits,
            "title": titles,
            "post": posts,
            "feedback": feedbacks,
            "generated_summary_for_feedback": original_summaries,
            "causal_response_ensemble_ordered_refinement_selection": ensemble_ordered_refinement_selection,
            "causal_response_ensemble_refinement_selection_log_probabilities": ensemble_refinement_selection_log_probabilities,
            "causal_response_ensemble_selected_refinement": ensemble_selected_refinements[
                "ensemble_selected_refinement"
            ],
        }
    )
    for top_n in range(2, select_top_n + 1):
        dataframe[
            "causal_response_ensemble_selected_refinement_top_{}".format(top_n)
        ] = ensemble_selected_refinements[
            "ensemble_selected_refinement_top_{}".format(top_n)
        ]
    dataframe.to_json(output_path, lines=True, orient="records")


if __name__ == "__main__":
    main()
