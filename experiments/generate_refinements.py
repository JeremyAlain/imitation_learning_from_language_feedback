import os.path
import string
from typing import List, Optional

import click
import pandas as pd
import pytorch_lightning as pl

from gpt3_language_model import GPT3LanguageModel
from utilities.summarization_utilities import (
    completions_to_postprocessed_completions_per_prompt,
    prepare_dataset,
    re_generate_degenerate_summaries,
)


@click.command()
@click.option(
    "--model_names", required=True, default=["text-davinci-001"], multiple=True,
)
@click.option(
    "--prompt_type",
    required=True,
    type=click.Choice(
        ["feedback_refinement", "direct_refinement"], case_sensitive=False
    ),
    default="feedback_refinement",
)
@click.option("--feedback_summary_name", required=True, type=str)
@click.option("--number_of_generations_per_prompt", required=True, default=5, type=int)
@click.option("--dataset_path", required=True, type=str)
@click.option("--number_of_log_probabilities", required=True, type=int, default=0)
@click.option("--max_samples", required=False, type=int)
@click.option("--experiment_name", required=True, type=str)
@click.option("--forbidden_texts", required=False, type=str, multiple=True)
@click.option("--max_number_of_sampling_tries_per_prompt", required=False, type=int)
@click.option("--number_of_dataset_intervals", required=False, type=int)
def main(
    model_names: List[str],
    prompt_type: str,
    feedback_summary_name: str,
    number_of_generations_per_prompt: int,
    dataset_path: str,
    number_of_log_probabilities: int,
    max_samples: Optional[int],
    experiment_name: str,
    forbidden_texts: Optional[List[str]],
    max_number_of_sampling_tries_per_prompt: Optional[int],
    number_of_dataset_intervals: Optional[int],
) -> None:
    pl.seed_everything(42)
    assert os.path.isfile(dataset_path)

    if max_number_of_sampling_tries_per_prompt is None:
        max_number_of_sampling_tries_per_prompt = number_of_generations_per_prompt

    data = pd.read_json(dataset_path, lines=True, orient="records")

    if prompt_type == "feedback_refinement":
        summarization_instructions = "Write an excellent summary that incorporates the feedback on the given summary and is better than the given summary.\n\n"
    elif prompt_type == "direct_refinement":
        summarization_instructions = (
            "Write an excellent summary that is better than the given summary.\n\n"
        )
    else:
        raise NotImplementedError()

    dataset = prepare_dataset(
        data,
        prompt_type=prompt_type,
        instructions=summarization_instructions,
        feedback_summary_name=feedback_summary_name,
    )
    if max_samples is not None:
        assert max_samples <= dataset.shape[0]
        dataset = dataset.iloc[:max_samples]
    for model_name in model_names:
        generate_refinements(
            model_name,
            dataset,
            number_of_generations_per_prompt=number_of_generations_per_prompt,
            number_of_log_probabilities=number_of_log_probabilities,
            experiment_name=experiment_name,
            forbidden_texts=forbidden_texts,
            max_number_of_sampling_tries_per_prompt=max_number_of_sampling_tries_per_prompt,
            number_of_dataset_intervals=number_of_dataset_intervals,
        )


def generate_refinements(
    model_name: str,
    dataset: pd.DataFrame,
    number_of_generations_per_prompt: int,
    number_of_log_probabilities: int,
    experiment_name: str,
    max_number_of_sampling_tries_per_prompt: Optional[int],
    forbidden_texts: Optional[List[str]] = None,
    number_of_dataset_intervals: Optional[int] = None,
) -> None:
    assert model_name == "text-davinci-001"
    model = GPT3LanguageModel(model_name=model_name)
    all_prompts = dataset["prompt"].tolist()
    dataset.rename(columns={"prompt": "refinement_prompt"}, inplace=True)

    temperature = 1.0
    top_p = 0.95
    alphabet = string.ascii_uppercase

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
            "Generating refinements for interval {} of {}".format(
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
            save_intermediate_results_path="data/results/refinements/{}_{}_intermediate_results.jsonl".format(
                experiment_name, dataset.shape[0]
            ),
        )
        assert (
            len(completions) == len(interval_prompts) * number_of_generations_per_prompt
        )
        (
            refinements_per_prompt,
            log_probabilities_per_prompt,
        ) = completions_to_postprocessed_completions_per_prompt(
            completions,
            number_of_log_probabilities=number_of_log_probabilities,
            number_of_generations_per_prompt=number_of_generations_per_prompt,
            number_of_prompts=len(interval_prompts),
            remove_text_after_new_line=True,
        )

        assert isinstance(max_number_of_sampling_tries_per_prompt, int)
        if max_number_of_sampling_tries_per_prompt > number_of_generations_per_prompt:
            (
                refinements_per_prompt,
                log_probabilities_per_prompt,
            ) = re_generate_degenerate_summaries(
                interval_prompts,
                refinements_per_prompt,
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
                        "generated_refinement_{}_log_probabilities".format(alphabet[i])
                    ] = ""
                generated_summaries_log_probabilities = [
                    log_probabilities[i]
                    for log_probabilities in log_probabilities_per_prompt
                ]
                dataset[
                    "generated_refinement_{}_log_probabilities".format(alphabet[i])
                ].iloc[
                    current_interval_start_index:current_interval_end_index
                ] = generated_summaries_log_probabilities

            if current_interval_start_index == 0:
                dataset["generated_refinement_{}".format(alphabet[i])] = ""
            generated_summaries = [summary[i] for summary in refinements_per_prompt]
            dataset["generated_refinement_{}".format(alphabet[i])].iloc[
                current_interval_start_index:current_interval_end_index
            ] = generated_summaries

        dataset.to_json(
            "data/results/refinements/{}_{}.jsonl".format(
                experiment_name, dataset.shape[0]
            ),
            lines=True,
            orient="records",
        )
    os.remove(
        "data/results/refinements/{}_{}_intermediate_results.jsonl".format(
            experiment_name, dataset.shape[0]
        )
    )


if __name__ == "__main__":
    main()
