import logging
import os
import string
from typing import List, Tuple

import click
import pandas as pd
from transformers import GPT2Tokenizer

from gpt3_language_model import GPT3LanguageModel
from utilities.summarization_utilities import (
    completions_to_postprocessed_completions_per_prompt,
    generate_feedback_refinement_prompt_and_completion,
    post_process_summaries,
)


@click.command()
@click.option(
    "--experiment_name",
    type=str,
    required=True,
    default="feedback_refinement_finetuned_evaluation",
)
@click.option("--number_of_generations_per_prompt", type=int, required=True, default=1)
def main(experiment_name: str, number_of_generations_per_prompt: int = 1) -> None:
    print("Number of generations per prompt", number_of_generations_per_prompt)
    model_name = "-"
    test_dataset = pd.read_json("data/final_dataset/test_dataset_700.jsonl", lines=True)
    reward_model_selection_data = pd.read_json(
        "results_and_plots/results/64_summaries_for_reward_model_test_700_5k_model_with_ordered_summary_selection.jsonl",
        lines=True,
    )
    text_davinci_001_summaries = reward_model_selection_data[
        "generated_summary_0"
    ].tolist()
    assert test_dataset["id"].tolist() == reward_model_selection_data["id"].tolist()
    assert test_dataset["post"].tolist() == reward_model_selection_data["post"].tolist()
    assert (
        test_dataset["subreddit"].tolist()
        == reward_model_selection_data["subreddit"].tolist()
    )

    prompts = []
    for i in range(test_dataset.shape[0]):
        sample = test_dataset.iloc[i]
        initial_summary = text_davinci_001_summaries[i]
        prompt, _ = generate_feedback_refinement_prompt_and_completion(
            sample["title"],
            sample["post"],
            initial_summary,
            feedback=" ",
            refinement=" ",
        )
        prompts.append(prompt)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT3LanguageModel(model_name=model_name)
    temperature = 1.0
    top_p = 0.95
    alphabet = string.ascii_uppercase

    completions = model.generate_completion(
        prompts=prompts,
        batch_size=1,
        temperature=temperature,
        number_of_log_probabilities=0,
        max_tokens_to_generate=300,
        echo=False,
        top_p=top_p,
        number_of_generations_per_prompt=number_of_generations_per_prompt,
        presence_penalty=0,
        frequency_penalty=0,
        stop_words=["\n###"],
    )
    assert len(completions) == len(prompts) * number_of_generations_per_prompt
    (
        feedback_and_refinement_per_prompt,
        _,
    ) = completions_to_postprocessed_completions_per_prompt(
        completions,
        number_of_generations_per_prompt=number_of_generations_per_prompt,
        number_of_prompts=len(prompts),
        number_of_log_probabilities=0,
        remove_text_after_new_line=True,
        do_postprocessing=False,
    )
    assert len(feedback_and_refinement_per_prompt) == len(prompts)

    (
        generated_feedback_per_prompt,
        generated_refinements_per_prompt,
    ) = split_completion_into_feedback_and_refinement(
        feedback_and_refinement_per_prompt, tokenizer
    )

    for i in range(number_of_generations_per_prompt):
        generated_refinements_index_i = [
            refinement[i] for refinement in generated_refinements_per_prompt
        ]
        test_dataset[
            "generated_test_refinement_{}".format(alphabet[i])
        ] = generated_refinements_index_i

        generated_feedback_index_i = [
            feedback[i] for feedback in generated_feedback_per_prompt
        ]
        test_dataset[
            "generated_test_feedback_{}".format(alphabet[i])
        ] = generated_feedback_index_i

    test_dataset.to_json(
        "data/results/test_summaries/{}_{}_test_evaluation_feedback_and_refinements.jsonl".format(
            experiment_name, test_dataset.shape[0]
        ),
        lines=True,
        orient="records",
        force_ascii=False,
    )
    if os.path.isfile(
        "data/results/test_summaries/{}_{}_intermediate_test_results.jsonl".format(
            experiment_name, test_dataset.shape[0]
        )
    ):
        os.remove(
            "data/results/test_summaries/{}_{}_intermediate_test_results.jsonl".format(
                experiment_name, test_dataset.shape[0]
            ),
        )


def split_completion_into_feedback_and_refinement(
    feedback_and_refinement_list: List[List[str]], tokenizer: GPT2Tokenizer
) -> Tuple[List[List[str]], List[List[str]]]:
    all_feedback, all_refinements = [], []
    for i, feedback_and_refinements_per_prompt in enumerate(
        feedback_and_refinement_list
    ):
        feedback_per_prompt, refinements_per_prompt = [], []
        for j, feedback_and_refinement in enumerate(
            feedback_and_refinements_per_prompt
        ):
            if "\n\nImproved TL;DR:" in feedback_and_refinement:
                (feedback, refinement) = feedback_and_refinement.split(
                    "\n\nImproved TL;DR:"
                )
                feedback_per_prompt.append(feedback)
                refinements_per_prompt.append(refinement)
            else:
                logging.warning(
                    "No feedback and refinement found in sample {}, generation {} completion: {}".format(
                        i, j, feedback_and_refinement
                    )
                )
                feedback_per_prompt.append(feedback_and_refinement)
                refinements_per_prompt.append(feedback_and_refinement)
        assert len(feedback_per_prompt) == len(refinements_per_prompt)
        # feedback_per_prompt = post_process_summaries(feedback_per_prompt, remove_text_after_new_line=True)
        # note we need to already remove white spaces etc. before we tokenize. If we only remoe those after tokenization
        # the tokens will be again different and might increase to more than 48. So first remove incomplete text
        # and white spaces. Then remove tokens if we are above 48. Then postprocss again to make nice text.
        refinements_per_prompt = post_process_summaries(
            refinements_per_prompt, remove_text_after_new_line=True
        )
        refinements_per_prompt = shorten_refinements(refinements_per_prompt, tokenizer)
        refinements_per_prompt = post_process_summaries(
            refinements_per_prompt, remove_text_after_new_line=True
        )
        for refinement in refinements_per_prompt:
            assert (
                len(tokenizer.encode(refinement, add_special_tokens=False)) <= 48
            ), "Refinement too long: {}".format(refinement)
        all_feedback.append(feedback_per_prompt)
        all_refinements.append(refinements_per_prompt)
    assert (
        len(all_feedback) == len(all_refinements) == len(feedback_and_refinement_list)
    )
    return all_feedback, all_refinements


def shorten_refinements(
    refinements_per_prompt: List[str], tokenizer: GPT2Tokenizer
) -> List[str]:
    max_token_length = 48
    shortened_refinements = []
    for refinement in refinements_per_prompt:
        tokens = tokenizer.encode(refinement, add_special_tokens=False)
        if len(tokens) > max_token_length:
            tokens = tokens[:max_token_length]
            new_refinement = tokenizer.decode(tokens)
            shortened_refinements.append(new_refinement)
        else:
            shortened_refinements.append(refinement)
    return shortened_refinements


if __name__ == "__main__":
    main()
