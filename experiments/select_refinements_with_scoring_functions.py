import logging
import os.path
from typing import List, Optional

import click
import pandas as pd
import pytorch_lightning as pl

from scoring_functions.causal_response import CausalResponseScoring
from scoring_functions.embedding_similarity import (
    FeedbackRefinementEmbeddingSimilarityScoring,
)
from scoring_functions.log_probability import LogProbabilityScoring
from scoring_functions.random_selection import RandomScoring

all_scoring_functions = [
    "random",
    "embedding_similarity",
    "causal_response",
    "log_probability_scoring",
]


@click.command()
@click.option(
    "--scoring_function_names",
    required=True,
    multiple=True,
    type=click.Choice(all_scoring_functions, case_sensitive=False),
    default=all_scoring_functions,
)
@click.option("--dataset_path", required=True, type=str)
@click.option("--number_of_refinements", required=True, type=int)
@click.option("--model_name", required=False, type=str)
@click.option(
    "--feedback_summary_name",
    required=False,
    type=str,
    help="Name of the column of the summary on which the feedback was written.",
)
@click.option(
    "--probability_scoring_methods",
    required=False,
    multiple=True,
    default=[
        "ranking_r_given_xyf",
        "ranking_r_given_x",
        "ranking_f_given_xr",
        "ranking_f_given_xr_minus_expected_feedback",
        "ranking_f_given_xr_minus_r_given_xyf",
        "ranking_f_given_xr_minus_r_given_x",
        "ranking_f_given_xr_minus_r_given_xyf_minus_expected_feedback",
        "ranking_f_given_xr_minus_r_given_x_minus_expected_feedback",
        "ranking_joint_probability_xrf",
        "ranking_joint_probability_xyfr",
    ],
)
@click.option("--experiment_id", required=False, type=str)
@click.option(
    "--causal_response_prompt_type",
    required=False,
    type=click.Choice(
        ["prompt_1", "prompt_2", "prompt_3", "prompt_4", "prompt_5"],
        case_sensitive=False,
    ),
)
@click.option("--number_of_dataset_intervals", required=False, type=int)
def main(
    scoring_function_names: List[str],
    dataset_path: str,
    number_of_refinements: int,
    model_name: Optional[str],
    probability_scoring_methods: Optional[List[str]],
    experiment_id: Optional[str],
    feedback_summary_name: Optional[str],
    causal_response_prompt_type: Optional[str],
    number_of_dataset_intervals: Optional[int],
) -> None:
    assert os.path.isfile(dataset_path)
    dataset = pd.read_json(dataset_path, lines=True)
    output_folder = os.path.join(
        os.path.split(dataset_path)[0], "results_with_selected_refinements"
    )
    additional_results_output_folder = os.path.join(output_folder, "additional_results")

    for scoring_function_name in scoring_function_names:
        # seeding here so order of scoring function execution has no influence
        pl.seed_everything(42)
        logging.warning("Selecting refinements with {}".format(scoring_function_name))

        dataset_size = dataset.shape[0]
        if number_of_dataset_intervals is not None:
            all_interval_ranges = []
            dataset_interval_size = dataset_size // number_of_dataset_intervals
            for interval_index in range(number_of_dataset_intervals):
                start_index = interval_index * dataset_interval_size
                end_index = (interval_index + 1) * dataset_interval_size
                if interval_index == number_of_dataset_intervals - 1:
                    end_index = dataset_size
                all_interval_ranges.append((start_index, end_index))
        else:
            all_interval_ranges = [(0, dataset_size)]
            dataset_interval_size = len(all_interval_ranges)
            number_of_dataset_intervals = 1

        print(
            "Using {} intervals of size {}".format(
                len(all_interval_ranges), dataset_interval_size
            )
        )
        assert len(all_interval_ranges) == number_of_dataset_intervals

        all_refinement_scores = None
        for interval_index, (interval_start, interval_end) in enumerate(
            all_interval_ranges
        ):
            print(
                "Selecting Refinements for interval {} of {}".format(
                    interval_index, number_of_dataset_intervals
                )
            )
            current_dataset = dataset.iloc[interval_start:interval_end]
            if scoring_function_name == "random":
                scoring_function = RandomScoring(
                    data=current_dataset,
                    number_of_refinements=number_of_refinements,
                    scoring_function_name=scoring_function_name,
                )
                (
                    selected_refinements,
                    refinement_scores,
                ) = scoring_function.select_refinement()
            elif scoring_function_name == "embedding_similarity":
                scoring_function = FeedbackRefinementEmbeddingSimilarityScoring(
                    data=current_dataset,
                    number_of_refinements=number_of_refinements,
                    scoring_function_name=scoring_function_name,
                )
                (
                    selected_refinements,
                    refinement_scores,
                ) = scoring_function.select_refinement(
                    model_name="text-similarity-davinci-001"
                )
            elif scoring_function_name == "causal_response":
                assert model_name == "text-davinci-001"
                scoring_function = CausalResponseScoring(
                    data=current_dataset,
                    number_of_refinements=number_of_refinements,
                    scoring_function_name=scoring_function_name,
                    prompt_type=causal_response_prompt_type,
                )
                (
                    selected_refinements,
                    refinement_scores,
                ) = scoring_function.select_refinement(
                    model_name=model_name, feedback_summary_name=feedback_summary_name
                )
            elif scoring_function_name == "log_probability_scoring":
                scoring_function = LogProbabilityScoring(
                    data=current_dataset,
                    number_of_refinements=number_of_refinements,
                    scoring_function_name=scoring_function_name,
                    scoring_probabilitiy_methods=probability_scoring_methods,
                )
                assert model_name == "text-davinci-001"
                (
                    selected_refinements,
                    refinement_scores,
                ) = scoring_function.select_refinement(
                    model_name=model_name, feedback_summary_name=feedback_summary_name
                )
            else:
                raise NotImplementedError()

            if refinement_scores is not None:
                assert (
                    selected_refinements.shape[0]
                    == refinement_scores.shape[0]
                    == current_dataset.shape[0]
                )
            else:
                assert selected_refinements.shape[0] == current_dataset.shape[0]

            if interval_start == 0:
                dataset = pd.concat([dataset, selected_refinements], axis=1)
            else:
                dataset[
                    scoring_function.scoring_function_name + "_selected_refinement"
                ].iloc[interval_start:interval_end] = selected_refinements[
                    scoring_function.scoring_function_name + "_selected_refinement"
                ]
                dataset[
                    scoring_function.scoring_function_name
                    + "_ordered_refinement_selection"
                ].iloc[interval_start:interval_end] = selected_refinements[
                    scoring_function.scoring_function_name
                    + "_ordered_refinement_selection"
                ]

            output_file_name = os.path.splitext(os.path.split(dataset_path)[1])[
                0
            ] + "_with_selected_refinements_{}.jsonl".format(experiment_id)

            dataset.to_json(
                os.path.join(output_folder, output_file_name),
                lines=True,
                orient="records",
            )
            if refinement_scores is not None:
                os.makedirs(additional_results_output_folder, exist_ok=True)
                if scoring_function_name == "causal_response":
                    assert isinstance(causal_response_prompt_type, str)
                    additional_result_file_name = (
                        scoring_function_name + "_" + causal_response_prompt_type
                    )
                else:
                    additional_result_file_name = scoring_function_name

                if interval_start == 0:
                    all_refinement_scores = refinement_scores
                else:
                    assert isinstance(all_refinement_scores, pd.DataFrame)
                    all_refinement_scores = pd.concat(
                        [all_refinement_scores, refinement_scores], axis=0
                    )
                    all_refinement_scores.reset_index(drop=True, inplace=True)

                all_refinement_scores.to_json(
                    os.path.join(
                        additional_results_output_folder,
                        additional_result_file_name + "_calculations.jsonl",
                    ),
                    lines=True,
                    orient="records",
                )


if __name__ == "__main__":
    main()
