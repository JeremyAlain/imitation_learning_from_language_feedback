import re
from typing import List

import click
import numpy as np
import pandas as pd

from gpt3_language_model import GPT3LanguageModel

model_id_dictionary = {
    "text-davinci-001": "text-davinci-001",
    "human_100_n_epochs_1": "-",
    "human_100_n_epochs_2": "-",
    "human_100_n_epochs_3": "-",
    "human_100_n_epochs_4": "-",
    "human_100_prompt_loss_weight_0": "-",
    "human_100_prompt_loss_weight_0_01": "-",
    "human_100_prompt_loss_weight_0_05": "-",
    "human_100_prompt_loss_weight_0_1": "-",
    "human_100_lr_0_02": "-",
    "human_100_lr_0_05": "-",
    "human_100_lr_0_1": "-",
    "human_100_lr_0_2": "-",
    "human_1K_n_epochs_1": "-",
    "human_1K_n_epochs_2": "-",
    "human_1K_n_epochs_3": "-",
    "human_1K_n_epochs_4": "-",
    "human_1K_prompt_loss_weight_0": "-",
    "human_1K_prompt_loss_weight_0_01": "-",
    "human_1K_prompt_loss_weight_0_05": "-",
    "human_1K_prompt_loss_weight_0_1": "-",
    "human_1K_lr_0_02": "-",
    "human_1K_lr_0_05": "-",
    "human_1K_lr_0_1": "-",
    "human_1K_lr_0_2": "-",
    "human_5K_n_epochs_1": "-",
    "human_5K_n_epochs_2": "-",
    "human_5K_n_epochs_3": "-",
    "human_5K_n_epochs_4": "-",
    "human_5K_prompt_loss_weight_0": "-",
    "human_5K_prompt_loss_weight_0_01": "-",
    "human_5K_prompt_loss_weight_0_05": "-",
    "human_5K_prompt_loss_weight_0_1": "-",
    "human_5K_lr_0_02": "-",
    "human_5K_lr_0_05": "-",
    "human_5K_lr_0_1": "-",
    "human_5K_lr_0_2": "-",
    "refinements_200_n_epochs_1": "-",
    "refinements_200_n_epochs_2": "-",
    "refinements_200_n_epochs_3": "-",
    "refinements_200_n_epochs_4": "-",
    "refinements_200_prompt_loss_weight_0": "-",
    "refinements_200_prompt_loss_weight_0_01": "-",
    "refinements_200_prompt_loss_weight_0_05": "-",
    "refinements_200_prompt_loss_weight_0_1": "-",
    "refinements_200_lr_0_02": "-",
    "refinements_200_lr_0_05": "-",
    "refinements_200_lr_0_1": "-",
    "refinements_200_lr_0_2": "-",
    "refinements_300_n_epochs_1": "-",
    "refinements_300_n_epochs_2": "-",
    "refinements_300_n_epochs_3": "-",
    "refinements_300_n_epochs_4": "-",
    "refinements_300_prompt_loss_weight_0": "-",
    "refinements_300_prompt_loss_weight_0_01": "-",
    "refinements_300_prompt_loss_weight_0_05": "-",
    "refinements_300_prompt_loss_weight_0_1": "-",
    "refinements_300_lr_0_02": "-",
    "refinements_300_lr_0_05": "-",
    "refinements_300_lr_0_1": "-",
    "refinements_300_lr_0_2": "-",
}


@click.command()
@click.option(
    "--model_names", required=True, default=["text-davinci-001"], multiple=True,
)
@click.option("--dataset_type", required=True, default="human", type=str)
def main(model_names: List[str], dataset_type: str) -> None:
    if dataset_type == "human":
        data = pd.read_json(
            "data/finetuning_datasets/ideal_human_summary_finetuning_dataset_development_200.jsonl",
            lines=True,
        )
    elif dataset_type == "refinements":
        data = pd.read_json(
            "data/finetuning_datasets/selected_refinement_finetuning_dataset_top_1_development_200.jsonl",
            lines=True,
        )
    else:
        raise ValueError("Invalid dataset type")
    for model_name in model_names:
        assert dataset_type in model_name
    output_folder = "results_and_plots/results/perplexities/"
    prompts = (data["prompt"] + data["completion"]).tolist()
    for model_name in model_names:
        perplexity_per_sample = []
        assert model_name in list(model_id_dictionary.keys())
        model = GPT3LanguageModel(model_name=model_id_dictionary[model_name])
        completions = model.generate_completion(
            prompts=prompts,
            batch_size=1,
            temperature=1,
            number_of_log_probabilities=1,
            max_tokens_to_generate=0,
            echo=True,
            top_p=1.0,
            number_of_generations_per_prompt=1,
            presence_penalty=0,
            frequency_penalty=0,
        )
        for completion in completions:
            assert isinstance(completion, dict)
            # don't take none prob
            text = completion["text"]
            summary = re.search("TL;DR:(.+)", text).group(1)  # type: ignore
            text_start_index = text.rfind(summary)
            probability_start_index = completion["log_probabilities"][
                "text_offset"
            ].index(text_start_index)
            token_logprobs = completion["log_probabilities"]["token_logprobs"][
                probability_start_index:-1
            ]
            token_logprobs = [
                logprob for logprob in token_logprobs if logprob is not None
            ]
            # calculate perplexity
            perplexity = np.exp(-np.mean(token_logprobs))
            perplexity_per_sample.append(perplexity)
        average_perplexity = np.mean(perplexity_per_sample)
        print(f"Model: {model_name}, perplexity: {average_perplexity}")
        result = pd.DataFrame(
            {
                "model_name": model_name,
                "perplexity": average_perplexity,
                "all_perplexities": perplexity_per_sample,
            }
        )
        result.to_json(
            output_folder + model_name + "_perplexity.jsonl",
            orient="records",
            lines=True,
        )


if __name__ == "__main__":
    main()
