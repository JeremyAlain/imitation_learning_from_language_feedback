import os
import string
from typing import List, Optional

import click
import numpy as np
import pandas as pd

from gpt3_language_model import GPT3LanguageModel
from utilities.summarization_utilities import (
    completions_to_postprocessed_completions_per_prompt,
    generate_summary_prompt_format,
)

summarization_instruction = "Write an excellent summary of the given text.\n\n"
models_to_evaluate_with_id = {
    "text-davinci-001": "text-davinci-001",
    "finetuned_on_100_refinements": "-",
    "finetuned_on_1000_refinements": "-",
    "finetuned_on_5000_refinements": "-",
    "finetuned_on_100_human_summaries": "-",
    "finetuned_on_1000_human_summaries": "-",
    "finetuned_on_5000_human_summaries": "-",
    "finetuned_on_100_initial_summaries": "-",
    "finetuned_on_1000_initial_summaries": "-",
    "finetuned_on_5000_initial_summaries": "-",
    "iterative_finetuning_on_refinements_round_1": "-",
    "iterative_finetuning_on_refinements_round_2": "-",
    "refinement_baseline_round_1_200": "-",
    "refinement_baseline_round_2_300": "-",
    "iterative_finetuning_on_refinements_round_1_200": "-",
    "iterative_finetuning_on_refinements_round_2_300": "-",
    "refinement_baseline_round_1": "-",
    "refinement_baseline_round_2": "-",
}


@click.command()
@click.option(
    "--models_to_evaluate", required=True, default=["text-davinci-001"], multiple=True,
)
@click.option("--number_of_generations_per_prompt", type=int, required=True, default=1)
@click.option("--is_test_data", type=bool, required=True, default=True)
@click.option("--evaluation_dataset_path", type=str, required=False)
def main(
    models_to_evaluate: List[str],
    number_of_generations_per_prompt: int = 1,
    is_test_data: bool = True,
    evaluation_dataset_path: Optional[str] = None,
) -> None:
    for model_to_evaluate in models_to_evaluate:
        assert (
            model_to_evaluate in models_to_evaluate_with_id.keys()
        ), f"Model {model_to_evaluate} is not supported. Please choose from {models_to_evaluate_with_id.keys()}"

    if is_test_data:
        test_dataset = pd.read_json(
            "data/final_dataset/test_dataset_700.jsonl", lines=True
        )
    else:
        assert evaluation_dataset_path is not None
        # used to generate summaries for iterative refinement dataset
        test_dataset = pd.read_json(evaluation_dataset_path, lines=True)
    for model_to_evaluate in models_to_evaluate:
        print(
            "Evaluating model: {} with ID: {}".format(
                model_to_evaluate, models_to_evaluate_with_id[model_to_evaluate]
            )
        )
        test_prompts = create_test_prompts(test_dataset)
        model = GPT3LanguageModel(
            model_name=models_to_evaluate_with_id[model_to_evaluate]
        )
        temperature = 1.0
        top_p = 0.95
        alphabet = string.ascii_uppercase

        completions = model.generate_completion(
            prompts=test_prompts,
            batch_size=1,
            temperature=temperature,
            number_of_log_probabilities=1,
            max_tokens_to_generate=48,
            echo=False,
            top_p=top_p,
            number_of_generations_per_prompt=number_of_generations_per_prompt,
            presence_penalty=0,
            frequency_penalty=0,
            save_intermediate_results_path="data/results/summaries/{}_{}_intermediate_test_results.jsonl".format(
                model_to_evaluate, test_dataset.shape[0]
            ),
        )
        assert len(completions) == len(test_prompts) * number_of_generations_per_prompt
        (
            summaries_per_prompt,
            log_probabilities_per_prompt,
        ) = completions_to_postprocessed_completions_per_prompt(
            completions,
            number_of_generations_per_prompt=number_of_generations_per_prompt,
            number_of_prompts=len(test_prompts),
            number_of_log_probabilities=1,
            remove_text_after_new_line=True,
        )
        assert len(summaries_per_prompt) == len(test_prompts)
        for i in range(number_of_generations_per_prompt):
            generated_summaries_index_i = [
                summary[i] for summary in summaries_per_prompt
            ]
            test_dataset[
                "generated_test_summary_{}".format(alphabet[i])
            ] = generated_summaries_index_i

            log_probabilities_index_i = [
                log_probability[i] for log_probability in log_probabilities_per_prompt
            ]
            test_dataset[
                "log_probability_test_summary_{}".format(alphabet[i])
            ] = log_probabilities_index_i

            average_log_probabilitis_index_i = [
                np.mean(log_probability[i]["token_logprobs"])
                for log_probability in log_probabilities_per_prompt
            ]
            test_dataset[
                "average_log_probability_test_summary_{}".format(alphabet[i])
            ] = average_log_probabilitis_index_i

        if is_test_data:
            test_dataset.to_json(
                "data/results/test_summaries/{}_{}_test_evaluation_summaries.jsonl".format(
                    model_to_evaluate, test_dataset.shape[0]
                ),
                lines=True,
                orient="records",
                force_ascii=False,
            )
        else:
            assert evaluation_dataset_path is not None
            test_dataset.to_json(
                "data/results/test_summaries/{}_{}_{}_evaluation_summaries.jsonl".format(
                    model_to_evaluate,
                    test_dataset.shape[0],
                    evaluation_dataset_path.split("/")[-1].split(".")[0],
                ),
                lines=True,
                orient="records",
                force_ascii=False,
            )
        if os.path.isfile(
            "data/results/test_summaries/{}_{}_intermediate_test_results.jsonl".format(
                model_to_evaluate, test_dataset.shape[0]
            )
        ):
            os.remove(
                "data/results/test_summaries/{}_{}_intermediate_test_results.jsonl".format(
                    model_to_evaluate, test_dataset.shape[0]
                ),
            )


def create_test_prompts(dataset: pd.DataFrame) -> List[str]:
    prompts = []
    for i in range(dataset.shape[0]):
        sample = dataset.iloc[i]
        prompt = summarization_instruction + generate_summary_prompt_format(
            sample["title"], sample["post"]
        )
        prompts.append(prompt)
    return prompts


if __name__ == "__main__":
    main()
