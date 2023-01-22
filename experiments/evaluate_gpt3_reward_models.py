import click
import numpy as np
import pandas as pd

from gpt3_language_model import GPT3LanguageModel


@click.command()
@click.option(
    "--model_type",
    type=click.Choice(["comparison", "classification"], case_sensitive=False),
)
def main(model_type: str) -> None:
    if model_type == "comparison":
        model_name = (
            "-"
        )
        validation_data = pd.read_json(
            "data/finetuning_datasets/reward_model_comparison_finetuning_dataset_validation_500.jsonl",
            lines=True,
        )
        input_prompts = validation_data["prompt"].tolist()

    elif model_type == "classification":
        model_name = "-"
        validation_data = pd.read_json(
            "data/finetuning_datasets/reward_model_classification_finetuning_dataset_validation_1000.jsonl",
            lines=True,
        )
        input_prompts = validation_data["prompt"].tolist()
        for i in range(len(input_prompts) // 2):
            assert (
                input_prompts[2 * i].split("TL;DR")[0]
                == input_prompts[2 * i + 1].split("TL;DR")[0]
            )
    else:
        raise ValueError(
            f"model_type must be either 'comparison' or 'classification', not {model_type}"
        )

    temperature = 0
    top_p = 0
    model = GPT3LanguageModel(model_name=model_name)
    if model_type == "comparison":
        # Logit biases
        # token_id: 317 = " A"
        # token_id: 347 = " B"
        logit_bias = {"317": 100, "347": 100}
        predictions = model.generate_completion(
            prompts=input_prompts,
            batch_size=1,
            max_tokens_to_generate=1,
            number_of_log_probabilities=5,
            echo=False,
            top_p=top_p,
            temperature=temperature,
            number_of_generations_per_prompt=1,
            presence_penalty=0,
            frequency_penalty=0,
        )
        normalized_A_probabilities = []
        for prediction in predictions:
            assert isinstance(prediction, dict)
            A_log_probability = prediction["log_probabilities"]["top_logprobs"][0][" A"]
            B_log_probability = prediction["log_probabilities"]["top_logprobs"][0][" B"]

            normalized_A_probability = np.exp(A_log_probability) / (
                np.exp(A_log_probability) + np.exp(B_log_probability)
            )
            normalized_A_probabilities.append(normalized_A_probability)
        validation_data["normalized_A_probability"] = normalized_A_probabilities

        validation_data.to_json(
            "results_and_plots/results/reward_model_500_validation_comparison_dataset_predictions.jsonl",
            lines=True,
            orient="records",
        )

    elif model_type == "classification":
        # Logit biases
        # token_id: 1400 = " No"
        # token_id: 3363 = " Yes"
        logit_bias = {"3363": 100, "1400": 100}
        predictions = model.generate_completion(
            prompts=input_prompts,
            batch_size=1,
            max_tokens_to_generate=1,
            number_of_log_probabilities=2,
            echo=False,
            top_p=top_p,
            temperature=temperature,
            number_of_generations_per_prompt=1,
            presence_penalty=0,
            frequency_penalty=0,
            logit_bias=logit_bias,
        )
        normalized_yes_probabilities = []
        for prediction in predictions:
            assert isinstance(prediction, dict)
            positive_log_probability = prediction["log_probabilities"]["top_logprobs"][
                0
            ][" Yes"]
            negative_log_probability = prediction["log_probabilities"]["top_logprobs"][
                0
            ][" No"]

            normalized_yes_probability = np.exp(positive_log_probability) / (
                np.exp(positive_log_probability) + np.exp(negative_log_probability)
            )
            normalized_yes_probabilities.append(normalized_yes_probability)
        validation_data["normalized_yes_probability"] = normalized_yes_probabilities

        validation_data.to_json(
            "results_and_plots/results/reward_model_500_validation_classification_dataset_predictions.jsonl",
            lines=True,
            orient="records",
        )


if __name__ == "__main__":
    main()
