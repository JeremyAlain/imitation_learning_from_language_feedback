import string
from typing import Dict, List

import click
import numpy as np
import pandas as pd

from gpt3_language_model import GPT3LanguageModel

alphabet = string.ascii_uppercase
number_of_refinements = 5


@click.command()
@click.option(
    "--dataset",
    type=click.Choice(["select_best_refinement_dev", "select_best_summary_test"]),
)
def main(dataset: str) -> None:
    model_name = (
        "-"
    )
    if dataset == "select_best_refinement_dev":
        data = pd.read_json(
            "data/results/refinements/development_feedback_refinements_200_classification_prompts.jsonl",
            lines=True,
        )
        assert data.shape[0] == 200
    elif dataset == "select_best_summary_test":
        data = pd.read_json(
            "data/results/test_summaries/finetuned_on_5000_refinements_700_test_evaluation_summaries_classificaiton_prompts.jsonl",
            lines=True,
        )
        assert data.shape[0] == 700

    prompts = []
    for i in range(data.shape[0]):
        for refinement_index in range(number_of_refinements):
            prompts.append(
                data.iloc[i][
                    "classification_prompt_summary_{}".format(
                        alphabet[refinement_index]
                    )
                ]
            )

    temperature = 0
    top_p = 0
    model = GPT3LanguageModel(model_name=model_name)

    # Logit biases
    # token_id: 1400 = " No"
    # token_id: 3363 = " Yes"
    logit_bias = {"3363": 100, "1400": 100}
    predictions = model.generate_completion(
        prompts=prompts,
        batch_size=1,
        max_tokens_to_generate=1,
        number_of_log_probabilities=5,
        echo=False,
        top_p=top_p,
        temperature=temperature,
        number_of_generations_per_prompt=1,
        presence_penalty=0,
        frequency_penalty=0,
        logit_bias=logit_bias,
    )
    assert len(predictions) == len(prompts) == number_of_refinements * data.shape[0]

    normalized_yes_probabilities = []
    for prediction in predictions:
        assert isinstance(prediction, dict)
        positive_log_probability = prediction["log_probabilities"]["top_logprobs"][0][
            " Yes"
        ]
        negative_log_probability = prediction["log_probabilities"]["top_logprobs"][0][
            " No"
        ]

        normalized_yes_probability = np.exp(positive_log_probability) / (
            np.exp(positive_log_probability) + np.exp(negative_log_probability)
        )
        normalized_yes_probabilities.append(normalized_yes_probability)

    scores_per_refinement: Dict[str, List[float]] = {}
    selected_refinements: List[str] = []
    gpt3_reward_model_ordered_refinement_selections = []
    if dataset == "select_best_refinement_dev":
        summary_name = "refinement"
    elif dataset == "select_best_summary_test":
        summary_name = "test_summary"
    else:
        raise ValueError("Invalid dataset")

    for refinement_index in range(number_of_refinements):
        scores_per_refinement[
            "generated_{}_{}_score".format(summary_name, alphabet[refinement_index])
        ] = []

    for i in range(data.shape[0]):
        current_scores = normalized_yes_probabilities[
            i * number_of_refinements : (i + 1) * number_of_refinements
        ]
        for refinement_index, score in enumerate(current_scores):
            scores_per_refinement[
                "generated_{}_{}_score".format(summary_name, alphabet[refinement_index])
            ].append(score)
        selected_refinement_indices_sorted = np.argsort(current_scores)[::-1]
        refinement_selection_letter = alphabet[selected_refinement_indices_sorted[0]]
        selected_refinements.append(
            data.iloc[i][
                "generated_{}_{}".format(summary_name, refinement_selection_letter)
            ]
        )
        refinement_selection_letters = [
            alphabet[index] for index in selected_refinement_indices_sorted
        ]
        gpt3_reward_model_ordered_refinement_selections.append(
            refinement_selection_letters
        )

    data[
        "gpt3_reward_model_ordered_{}_selections".format(summary_name)
    ] = gpt3_reward_model_ordered_refinement_selections
    data["selected_{}".format(summary_name)] = selected_refinements
    for refinement_index in range(number_of_refinements):
        data[
            "generated_{}_{}_score".format(summary_name, alphabet[refinement_index])
        ] = scores_per_refinement[
            "generated_refinement_{}_score".format(alphabet[refinement_index])
        ]

    if dataset == "select_best_refinement_dev":
        data.to_json(
            "data/results/refinements/results_with_selected_refinements/development/development_feedback_refinements_200_with_selected_refinements_gpt3_classification_reward_model.jsonl",
            lines=True,
            orient="records",
        )
    elif dataset == "select_best_summary_test":
        data.to_json(
            "data/results/refinements/results_with_selected_refinements/test/test_summaries_700_with_selected_refinements_gpt3_classification_reward_model.jsonl",
            lines=True,
            orient="records",
        )
    else:
        raise ValueError("Invalid dataset")


if __name__ == "__main__":
    main()
