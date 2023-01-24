import os
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from gpt3_language_model import GPT3LanguageModel

summarization_instruction = "Write an excellent summary of the given text.\n\n"


def main() -> None:
    initial_summary_finetuning_data = pd.read_json(
        "data/finetuning_datasets/initial_summary_finetuning_dataset_train_1000.jsonl",
        lines=True,
    )
    refinement_finetuning_daa = pd.read_json(
        "data/finetuning_datasets/selected_refinement_finetuning_dataset_top_1_train_1000.jsonl",
        lines=True,
    )
    human_summary_finetuning_data = pd.read_json(
        "data/finetuning_datasets/ideal_human_summary_finetuning_dataset_train_1000.jsonl",
        lines=True,
    )

    data_dict = {
        "initial_summary": initial_summary_finetuning_data,
        "refinement": refinement_finetuning_daa,
        "human_summary": human_summary_finetuning_data,
    }
    for dataset_name, dataset in data_dict.items():
        print(f"Calculating log probabilities for {dataset_name} data")
        prompts = (dataset["prompt"] + dataset["completion"]).tolist()
        model = GPT3LanguageModel(model_name="davinci")
        temperature = 0
        top_p = 0

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
        dataset["log_probabilitities_of_summary"] = log_probabilitity_of_summary
        dataset[
            "average_log_probability_of_summary"
        ] = average_log_probability_of_summary
        all_log_probs = []
        text_offsets = []
        tokens = []
        assert isinstance(completions, List)
        for i in range(len(completions)):
            assert isinstance(completions[i], dict)
            all_log_probs.append(completions[i]["log_probabilities"]["token_logprobs"])  # type: ignore
            text_offsets.append(completions[i]["log_probabilities"]["text_offset"])  # type: ignore
            tokens.append(completions[i]["log_probabilities"]["tokens"])  # type: ignore
        dataset["all_log_probs"] = all_log_probs
        dataset["text_offsets"] = text_offsets
        dataset["tokens"] = tokens
        dataset.to_json(
            os.path.join(
                "data/results/logprobs_of_train_data",
                "davinci_logprob_{}_1000.jsonl".format(dataset_name),
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
