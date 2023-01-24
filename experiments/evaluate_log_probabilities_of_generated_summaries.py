import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl

from gpt3_language_model import GPT3LanguageModel
from utilities.summarization_utilities import prepare_dataset


def main() -> None:
    """We generated 64 summaries with text-davinci-001 and used the reward model to select the best summary. An example
    results file is the following results_and_plots/results/64_summaries_for_reward_model_test_700_1k_model.jsonl . We did this with 3
    reward models which each picked a different summary. However, we forgot to calculate the log-probs of those summaries. this script
    will re-evaluate those log probs. Also, the first summary of those files is selected as a text-davinci-001 summary."""

    summaries_with_rm_selection_100 = pd.read_json(
        "results_and_plots/results/64_summaries_for_reward_model_test_700_100_model_with_ordered_summary_selection.jsonl",
        lines=True,
    )
    summaries_with_rm_selection_1k = pd.read_json(
        "results_and_plots/results/64_summaries_for_reward_model_test_700_1k_model_with_ordered_summary_selection.jsonl",
        lines=True,
    )
    summaries_with_rm_selection_5k = pd.read_json(
        "results_and_plots/results/64_summaries_for_reward_model_test_700_5k_model_with_ordered_summary_selection.jsonl",
        lines=True,
    )
    assert (
        summaries_with_rm_selection_1k["id"].tolist()
        == summaries_with_rm_selection_100["id"].tolist()
        == summaries_with_rm_selection_5k["id"].tolist()
    )
    assert (
        summaries_with_rm_selection_1k["ideal_human_summary"].tolist()
        == summaries_with_rm_selection_100["ideal_human_summary"].tolist()
        == summaries_with_rm_selection_5k["ideal_human_summary"].tolist()
    )
    number_of_generations_per_prompt = 64
    for i in range(number_of_generations_per_prompt):
        assert (
            summaries_with_rm_selection_1k["generated_summary_{}".format(i)].tolist()
            == summaries_with_rm_selection_100[
                "generated_summary_{}".format(i)
            ].tolist()
            == summaries_with_rm_selection_5k["generated_summary_{}".format(i)].tolist()
        )

    evaluate_log_probs_of_summaries(
        summaries_with_rm_selection_100,
        "generated_summary_0",
        "results_and_plots/results/64_summaries_for_reward_model_test_700_100_model_with_ordered_summary_selection_and_post_processed_summary_0_log_probs.jsonl",
    )
    evaluate_log_probs_of_summaries(
        summaries_with_rm_selection_100,
        "opt_reward_model_selected_summary",
        "results_and_plots/results/64_summaries_for_reward_model_test_700_100_model_with_ordered_summary_selection_and_post_processed_RM_log_probs.jsonl",
    )
    evaluate_log_probs_of_summaries(
        summaries_with_rm_selection_1k,
        "opt_reward_model_selected_summary",
        "results_and_plots/results/64_summaries_for_reward_model_test_700_1K_model_with_ordered_summary_selection_and_post_processed_RM_log_probs.jsonl",
    )
    evaluate_log_probs_of_summaries(
        summaries_with_rm_selection_5k,
        "opt_reward_model_selected_summary",
        "results_and_plots/results/64_summaries_for_reward_model_test_700_5K_model_with_ordered_summary_selection_and_post_processed_RM_log_probs.jsonl",
    )


def evaluate_log_probs_of_summaries(
    dataset: pd.DataFrame, summary_column_name: str, output_file_path: str
) -> None:
    pl.seed_everything(42)
    summarization_instructions = "Write an excellent summary of the given text.\n\n"
    dataset = prepare_dataset(
        dataset, prompt_type="summary", instructions=summarization_instructions
    )
    all_prompts = dataset["prompt"].tolist()
    dataset.drop(columns=["prompt"], inplace=True)
    prompts_and_completions = []
    for i, prompt in enumerate(all_prompts):
        summary = dataset[summary_column_name].iloc[i]
        prompt_and_completion = prompt + " " + summary
        prompts_and_completions.append(prompt_and_completion)

    assert len(all_prompts) == 698
    model = GPT3LanguageModel(model_name="text-davinci-001")

    temperature = 1.0
    top_p = 0.95
    completions = model.generate_completion(
        prompts=prompts_and_completions,
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
    assert len(completions) == len(all_prompts) == len(prompts_and_completions)
    (
        log_probabilitity_of_summary,
        average_log_probability_of_summary,
    ) = calculate_log_probabilities_of_summaries(
        completions  # type: ignore
    )
    dataset["log_probabilitity_of_postprocessed_summary"] = log_probabilitity_of_summary
    dataset[
        "average_log_probability_of_postprocessed_summary"
    ] = average_log_probability_of_summary
    dataset.to_json(output_file_path, orient="records", lines=True)


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
