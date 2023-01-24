import os
import re
from typing import Any, Dict, List, Tuple

import click
import numpy as np
import pandas as pd

from gpt3_language_model import GPT3LanguageModel

summarization_instruction = "Write an excellent summary of the given text.\n\n"
models_to_evaluate_with_id = {
    "finetuned_on_100_refinements": "-",
    "finetuned_on_1000_refinements": "-",
    "finetuned_on_5000_refinements": "-",
    "finetuned_on_100_human_summaries": "-",
    "finetuned_on_1000_human_summaries": "-",
    "finetuned_on_5000_human_summaries": "-",
    "finetuned_on_100_initial_summaries": "",
    "finetuned_on_1000_initial_summaries": "-",
    "finetuned_on_5000_initial_summaries": "-",
}


@click.command()
@click.option(
    "--models_to_evaluate",
    required=True,
    default=["finetuned_on_100_refinements"],
    multiple=True,
)
def main(models_to_evaluate: List[str]) -> None:
    for model_to_evaluate in models_to_evaluate:
        assert (
            model_to_evaluate in models_to_evaluate_with_id.keys()
        ), f"Model {model_to_evaluate} is not supported. Please choose from {models_to_evaluate_with_id.keys()}"

    for model_to_evaluate in models_to_evaluate:
        if (
            model_to_evaluate == "finetuned_on_100_refinements"
            or model_to_evaluate == "finetuned_on_1000_refinements"
            or model_to_evaluate == "finetuned_on_5000_refinements"
        ):
            dataset = pd.read_json(
                "data/finetuning_datasets/selected_refinement_finetuning_dataset_top_1_validation_500.jsonl",
                lines=True,
            )
        elif (
            model_to_evaluate == "finetuned_on_100_human_summaries"
            or model_to_evaluate == "finetuned_on_1000_human_summaries"
            or model_to_evaluate == "finetuned_on_5000_human_summaries"
        ):
            dataset = pd.read_json(
                "data/finetuning_datasets/ideal_human_summary_finetuning_dataset_validation_500.jsonl",
                lines=True,
            )
        elif (
            model_to_evaluate == "finetuned_on_100_initial_summaries"
            or model_to_evaluate == "finetuned_on_1000_initial_summaries"
            or model_to_evaluate == "finetuned_on_5000_initial_summaries"
        ):
            dataset = pd.read_json(
                "data/finetuning_datasets/initial_summary_finetuning_dataset_validation_500.jsonl",
                lines=True,
            )
        else:
            raise ValueError("Model not supported")

        prompts = (dataset["prompt"] + dataset["completion"]).tolist()
        print(
            "Evaluating model: {} with ID: {}".format(
                model_to_evaluate, models_to_evaluate_with_id[model_to_evaluate]
            )
        )
        model = GPT3LanguageModel(
            model_name=models_to_evaluate_with_id[model_to_evaluate]
        )
        temperature = 1.0
        top_p = 0.95

        completions = model.generate_completion(
            prompts=prompts,
            batch_size=1,
            temperature=temperature,
            number_of_log_probabilities=1,
            max_tokens_to_generate=0,
            echo=True,
            top_p=top_p,
            number_of_generations_per_prompt=1,
            presence_penalty=0,
            frequency_penalty=0,
        )
        assert len(completions) == len(prompts)
        (
            log_probabilitity_of_summary,
            average_log_probability_of_summary,
        ) = calculate_log_probabilities_of_summaries(
            completions  # type: ignore
        )
        dataset["log_probabilitity_of_summary"] = log_probabilitity_of_summary
        dataset[
            "average_log_probability_of_summary"
        ] = average_log_probability_of_summary
        dataset.to_json(
            os.path.join(
                "data/results/logprobs_of_validation_summaries",
                "{}_validation_logprob_500.json".format(model_to_evaluate),
            ),
            orient="records",
            lines=True,
        )


def calculate_log_probabilities_of_summaries(
    completions: List[Dict[str, Any]]
) -> Tuple[List[float], List[np.ndarray]]:
    log_probabilities = []
    average_log_probabilities = []
    for i in range(len(completions)):
        completion = completions[i]
        summary = re.search("TL;DR:(.+)", completion["text"]).group(1)  # type: ignore
        summary_start_index = completion["text"].rfind(summary)
        probability_start_index = completion["log_probabilities"]["text_offset"].index(
            summary_start_index
        )
        log_probabilities.append(
            completion["log_probabilities"]["token_logprobs"][probability_start_index:]
        )
        average_log_probabilities.append(
            np.mean(
                completion["log_probabilities"]["token_logprobs"][
                    probability_start_index:
                ]
            )
        )
    return log_probabilities, average_log_probabilities


if __name__ == "__main__":
    main()
