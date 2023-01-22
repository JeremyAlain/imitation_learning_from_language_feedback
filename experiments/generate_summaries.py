import os
import string
from typing import List, Optional

import click
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from gpt3_language_model import GPT3LanguageModel
from utilities.summarization_utilities import (
    completions_to_postprocessed_completions_per_prompt,
    prepare_dataset,
    re_generate_degenerate_summaries,
)


@click.command()
@click.option("--dataset_path", required=True, type=str)
@click.option("--number_of_generations_per_prompt", required=True, default=3, type=int)
@click.option("--number_of_log_probabilities", required=True, type=int, default=0)
@click.option("--max_samples", required=False, type=int)
@click.option("--experiment_name", required=True, type=str)
@click.option("--forbidden_texts", required=False, type=str, multiple=True)
@click.option("--max_number_of_sampling_tries_per_prompt", required=False, type=int)
@click.option("--number_of_dataset_intervals", required=False, type=int)
@click.option("--use_alphabet", required=False, type=bool, default=True)
def main(
    dataset_path: str,
    number_of_generations_per_prompt: int,
    number_of_log_probabilities: int,
    max_samples: Optional[int],
    experiment_name: str,
    forbidden_texts: Optional[List[str]],
    max_number_of_sampling_tries_per_prompt: Optional[int],
    number_of_dataset_intervals: Optional[int],
    use_alphabet: bool,
) -> None:
    pl.seed_everything(42)

    if max_number_of_sampling_tries_per_prompt is None:
        max_number_of_sampling_tries_per_prompt = number_of_generations_per_prompt

    data = pd.read_json(dataset_path, lines=True)
    summarization_instructions = "Write an excellent summary of the given text.\n\n"

    dataset = prepare_dataset(
        data, prompt_type="summary", instructions=summarization_instructions
    )
    if max_samples is not None:
        assert max_samples <= dataset.shape[0]
        dataset = dataset.iloc[:max_samples]

    generate_summaries(
        dataset,
        number_of_generations_per_prompt,
        number_of_log_probabilities,
        experiment_name,
        forbidden_texts=forbidden_texts,
        max_number_of_sampling_tries_per_prompt=max_number_of_sampling_tries_per_prompt,
        number_of_dataset_intervals=number_of_dataset_intervals,
        use_alphabet=use_alphabet,
    )


def generate_summaries(
    dataset: pd.DataFrame,
    number_of_generations_per_prompt: int,
    number_of_log_probabilities: int,
    experiment_name: str,
    max_number_of_sampling_tries_per_prompt: int,
    forbidden_texts: Optional[List[str]] = None,
    number_of_dataset_intervals: Optional[int] = None,
    use_alphabet: bool = True,
) -> None:
    model = GPT3LanguageModel(model_name="text-davinci-001")
    all_prompts = dataset["prompt"].tolist()
    dataset.rename(columns={"prompt": "summary_prompt"}, inplace=True)

    temperature = 1.0
    top_p = 0.95
    if use_alphabet:
        summary_identifiers = list(string.ascii_uppercase)
    else:
        summary_identifiers = [str(i) for i in range(number_of_generations_per_prompt)]

    if number_of_dataset_intervals is not None:
        # split the data into equally sized intervals. Then run everything on each interval and merge results together.
        # This is useful for running on many samples, and not having to wait for the whole thing to finish, and then break.
        all_interval_prompts = []
        all_interval_ranges = []
        dataset_size = dataset.shape[0]
        dataset_interval_size = dataset_size // number_of_dataset_intervals
        for interval_index in range(number_of_dataset_intervals):
            start_index = interval_index * dataset_interval_size
            end_index = (interval_index + 1) * dataset_interval_size
            if interval_index == number_of_dataset_intervals - 1:
                end_index = dataset_size
            interval_prompts = all_prompts[start_index:end_index]
            all_interval_prompts.append(interval_prompts)
            all_interval_ranges.append((start_index, end_index))
    else:
        all_interval_prompts = [all_prompts]
        all_interval_ranges = [(0, len(all_prompts))]
        dataset_interval_size = len(all_prompts)

    print(
        "Using {} intervals of size {}".format(
            len(all_interval_prompts), dataset_interval_size
        )
    )

    for interval_index, interval_prompts in enumerate(all_interval_prompts):
        print(
            "Generating summaries for interval {} of {}".format(
                interval_index, number_of_dataset_intervals
            )
        )
        completions = model.generate_completion(
            prompts=interval_prompts,
            batch_size=1,
            temperature=temperature,
            number_of_log_probabilities=number_of_log_probabilities,
            max_tokens_to_generate=48,
            echo=False,
            top_p=top_p,
            number_of_generations_per_prompt=number_of_generations_per_prompt,
            presence_penalty=0,
            frequency_penalty=0,
            save_intermediate_results_path="data/results/summaries/{}_{}_intermediate_results.jsonl".format(
                experiment_name, dataset.shape[0]
            ),
        )
        assert (
            len(completions) == len(interval_prompts) * number_of_generations_per_prompt
        )
        (
            summaries_per_prompt,
            log_probabilities_per_prompt,
        ) = completions_to_postprocessed_completions_per_prompt(
            completions,
            number_of_generations_per_prompt=number_of_generations_per_prompt,
            number_of_prompts=len(interval_prompts),
            number_of_log_probabilities=number_of_log_probabilities,
            remove_text_after_new_line=True,
        )
        if max_number_of_sampling_tries_per_prompt > number_of_generations_per_prompt:
            (
                summaries_per_prompt,
                log_probabilities_per_prompt,
            ) = re_generate_degenerate_summaries(
                interval_prompts,
                summaries_per_prompt,
                log_probabilities_per_prompt=log_probabilities_per_prompt,
                model=model,
                number_of_log_probabilities=number_of_log_probabilities,
                number_of_generations_per_prompt=number_of_generations_per_prompt,
                forbidden_texts=forbidden_texts,
                max_number_of_sampling_tries_per_prompt=max_number_of_sampling_tries_per_prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens_to_generate=48,
                presence_penalty=0,
                frequency_penalty=0,
                remove_text_after_new_line=True,
            )

        current_interval_start_index, current_interval_end_index = all_interval_ranges[
            interval_index
        ]
        for i in range(number_of_generations_per_prompt):
            if number_of_log_probabilities > 0:
                if current_interval_start_index == 0:
                    dataset[
                        "generated_summaries_{}_log_probabilities".format(
                            summary_identifiers[i]
                        )
                    ] = ""

                    dataset[
                        "average_log_probability_test_summary_{}".format(
                            summary_identifiers[i]
                        )
                    ] = ""

                generated_summaries_log_probabilities = [
                    log_probabilities[i]
                    for log_probabilities in log_probabilities_per_prompt
                ]
                dataset[
                    "generated_summaries_{}_log_probabilities".format(
                        summary_identifiers[i]
                    )
                ].iloc[
                    current_interval_start_index:current_interval_end_index
                ] = generated_summaries_log_probabilities

                average_log_probabilitis_index_i = [
                    np.mean(log_probability[i]["token_logprobs"])
                    for log_probability in log_probabilities_per_prompt
                ]
                dataset[
                    "average_log_probability_test_summary_{}".format(
                        summary_identifiers[i]
                    )
                ].iloc[
                    current_interval_start_index:current_interval_end_index
                ] = average_log_probabilitis_index_i

            generated_summaries = [summary[i] for summary in summaries_per_prompt]
            if current_interval_start_index == 0:
                dataset["generated_summary_{}".format(summary_identifiers[i])] = ""
            dataset["generated_summary_{}".format(summary_identifiers[i])].iloc[
                current_interval_start_index:current_interval_end_index
            ] = generated_summaries

        dataset.to_json(
            "data/results/summaries/{}_{}.jsonl".format(
                experiment_name, dataset.shape[0]
            ),
            lines=True,
            orient="records",
        )
        os.remove(
            "data/results/summaries/{}_{}_intermediate_results.jsonl".format(
                experiment_name, dataset.shape[0]
            )
        )


if __name__ == "__main__":
    main()
