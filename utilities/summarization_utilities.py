import re
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from gpt3_language_model import GPT3LanguageModel
from utilities.language_model_utilities import (
    list_of_completions_to_completions_per_prompt,
    texts_have_forbidden_substrings,
)


def prepare_dataset(
    dataset: pd.DataFrame,
    prompt_type: str,
    instructions: str,
    feedback_summary_name: Optional[str] = None,
) -> pd.DataFrame:
    assert "title" in dataset.columns
    assert "post" in dataset.columns

    list_of_dataset_prompts = []
    for i in range(dataset.shape[0]):
        sample = dataset.iloc[i]
        if prompt_type == "summary":
            prompt = generate_summary_prompt_format(sample["title"], sample["post"])
        elif prompt_type == "feedback_refinement":
            assert feedback_summary_name is not None
            assert feedback_summary_name in dataset.columns
            assert "feedback" in dataset.columns
            prompt = generate_feedback_refinement_prompt_format(
                sample["title"],
                sample["post"],
                sample[feedback_summary_name],
                sample["feedback"],
            )
        elif prompt_type == "direct_refinement":
            assert feedback_summary_name is not None
            assert feedback_summary_name in dataset.columns
            prompt = generate_general_refinement_prompt_format(
                sample["title"], sample["post"], sample[feedback_summary_name]
            )
        else:
            raise NotImplementedError()

        prompt_with_instruction = instructions + prompt
        list_of_dataset_prompts.append(prompt_with_instruction)
    dataset["prompt"] = list_of_dataset_prompts
    return dataset


def generate_summary_prompt_format(title: str, post: str) -> str:
    prompt = _title_post_format(title, post)
    prompt += "TL;DR:"
    return prompt


def generate_feedback_refinement_prompt_format(
    title: str, post: str, summary: str, feedback: str
) -> str:
    prompt = _title_post_format(title, post)
    prompt += "Summary: " + summary + "\n\n"
    prompt += "Feedback on summary: " + feedback + "\n\n"
    prompt += "Improved TL;DR:"
    return prompt


def generate_general_refinement_prompt_format(
    title: str, post: str, summary: str
) -> str:
    prompt = _title_post_format(title, post)
    prompt += "Summary: " + summary + "\n\n"
    prompt += "Improved TL;DR:"
    return prompt


def _title_post_format(title: str, post: str) -> str:
    prompt = "Title: " + title + "\n\n"
    prompt += "Text: " + post + "\n\n"
    return prompt


def refinement_given_summary_feedback_prompt_format(
    title: str, post: str, summary: str, feedback: str, refinement: str
) -> str:
    prompt = _title_post_format(title, post)
    prompt += "Summary: " + summary + "\n\n"
    prompt += "Feedback on summary: " + feedback + "\n\n"
    # We use two new lines at the end, in order to have a consistent structure. This is necessary to calculate
    # log-probabilities of certain ranges, so we always know where the text ends.
    prompt += "Improved TL;DR: " + refinement + "\n\n"
    return prompt


def feedback_given_refinement_prompt_format(
    title: str, post: str, refinement: str, feedback: str
) -> str:
    prompt = _title_post_format(title, post)
    prompt += "TL;DR: " + refinement + "\n\n"
    # We use two new lines at the end, in order to have a consistent structure. This is necessary to calculate
    # log-probabilities of certain ranges, so we always know where the text ends.
    prompt += "Feedback on summary: " + feedback + "\n\n"
    return prompt


def generate_reasoning_on_feedback_prompt_format(
    title: str, post: str, summary: str, feedback: str
) -> str:
    prompt = _title_post_format(title, post)
    prompt += "Summary: " + summary + "\n\n"
    prompt += "Feedback on summary: " + feedback + "\n\n"
    # prompt += "How can we best incorporate the feedback to write an improved summary, let's think step by step:"
    prompt += "How can you use the feedback to write an improved summary? Let's think step by step:"
    return prompt


def generate_refinement_from_reasoning_on_feedback_prompt_format(
    title: str, post: str, summary: str, feedback: str, reasoning: str
) -> str:
    prompt = _title_post_format(title, post)
    prompt += "Summary: " + summary + "\n\n"
    prompt += "Feedback on summary:\n" + feedback + "\n\n"
    prompt += (
        "How to use the feedback to write an improved summary? Let's think step by step:\n"
        + reasoning
        + "\n\n"
    )
    prompt += "Improved TL;DR:"
    return prompt


def post_process_summaries(
    list_of_summaries: List[str], remove_text_after_new_line: bool = True,
) -> List[str]:
    post_processed_summaries = []
    for summary in list_of_summaries:

        # remove empty spaces, newlines etc.. in front of text
        for i, letter in enumerate(summary):
            if letter.isalnum():
                summary = summary[i:]
                break

        # remove empty whitespace in the front, end, middle
        summary = re.sub(" +", " ", summary)
        summary = summary.strip()

        # in case there are new lines remove it and everything after that
        if remove_text_after_new_line and "\n" in summary:
            summary = summary.split("\n")[0]

        # Try to remove incomplete sentences. This is a heuristic that works most of the time, but not perfect.
        if "." in summary or "?" in summary or "!" in summary:
            summary = "".join(re.split(r"([.!?])", summary)[:-1])

        post_processed_summaries.append(summary)
    return post_processed_summaries


def completions_to_postprocessed_completions_per_prompt(
    completions: Union[List[str], List[Dict[str, Any]]],
    number_of_log_probabilities: int,
    number_of_generations_per_prompt: int,
    number_of_prompts: int,
    remove_text_after_new_line: bool,
    do_postprocessing: bool = True,
) -> Tuple[List[List[str]], Optional[List[List[Dict[str, Any]]]]]:
    assert len(completions) == number_of_prompts * number_of_generations_per_prompt
    generated_completions = []
    if number_of_log_probabilities > 0:
        assert isinstance(completions, List)
        log_probabilities = []

        for completion in completions:
            assert isinstance(completion, dict)
            generated_completions.append(completion["text"])
            log_probabilities.append(completion["log_probabilities"])
        assert len(generated_completions) == len(log_probabilities)

        log_probabilities_per_prompt = list_of_completions_to_completions_per_prompt(
            log_probabilities, number_of_prompts, number_of_generations_per_prompt
        )
    else:
        assert isinstance(completions, list)
        for completion in completions:
            assert isinstance(completion, str)
            generated_completions.append(completion)
        log_probabilities_per_prompt = None

    if do_postprocessing:
        generated_completions = post_process_summaries(
            generated_completions,
            remove_text_after_new_line=remove_text_after_new_line,
        )

    completions_per_prompt = list_of_completions_to_completions_per_prompt(
        generated_completions, number_of_prompts, number_of_generations_per_prompt
    )
    return completions_per_prompt, log_probabilities_per_prompt


def re_generate_degenerate_summaries(
    prompts: List[str],
    summaries_per_prompt: List[List[str]],
    log_probabilities_per_prompt: Optional[List[List[Dict[str, Any]]]],
    model: GPT3LanguageModel,
    number_of_log_probabilities: int,
    number_of_generations_per_prompt: int,
    forbidden_texts: Optional[List[str]],
    max_number_of_sampling_tries_per_prompt: int,
    temperature: float,
    max_tokens_to_generate: int,
    top_p: float,
    presence_penalty: float,
    frequency_penalty: float,
    remove_text_after_new_line: bool,
    stop_words: Optional[List[str]] = None,
) -> Tuple[List[List[str]], Optional[List[List[Dict[str, Any]]]]]:
    assert len(prompts) == len(summaries_per_prompt)
    if log_probabilities_per_prompt is not None:
        assert len(prompts) == len(log_probabilities_per_prompt)
    assert max_number_of_sampling_tries_per_prompt > number_of_generations_per_prompt
    indices_of_degenerate_summaries = []
    prompts_of_degenerate_summaries = []
    number_of_forbidden_text_summaries = 0
    for i, summaries in enumerate(summaries_per_prompt):
        if len(set(summaries)) != len(summaries) or texts_have_forbidden_substrings(
            summaries, forbidden_texts
        ):
            if texts_have_forbidden_substrings(summaries, forbidden_texts):
                number_of_forbidden_text_summaries += 1
            indices_of_degenerate_summaries.append(i)
            prompts_of_degenerate_summaries.append(prompts[i])

    print(
        "Regenerating {} degenerate summaries.".format(
            len(indices_of_degenerate_summaries)
        )
    )
    print(
        "{} degenerate summaries are due to forbidden texts.".format(
            number_of_forbidden_text_summaries
        )
    )
    if len(indices_of_degenerate_summaries) > 0:
        number_of_resampling_generations_per_prompt = (
            max_number_of_sampling_tries_per_prompt - number_of_generations_per_prompt
        )
        completions = model.generate_completion(
            prompts=prompts_of_degenerate_summaries,
            batch_size=1,
            temperature=temperature,
            number_of_log_probabilities=number_of_log_probabilities,
            max_tokens_to_generate=max_tokens_to_generate,
            echo=False,
            top_p=top_p,
            number_of_generations_per_prompt=number_of_resampling_generations_per_prompt,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            stop_words=stop_words,
        )

        assert len(completions) == len(prompts_of_degenerate_summaries) * (
            number_of_resampling_generations_per_prompt
        )
        (
            new_summaries_per_prompt,
            new_log_probabilities_per_prompt,
        ) = completions_to_postprocessed_completions_per_prompt(
            completions,
            number_of_log_probabilities=number_of_log_probabilities,
            number_of_generations_per_prompt=number_of_resampling_generations_per_prompt,
            number_of_prompts=len(prompts_of_degenerate_summaries),
            remove_text_after_new_line=remove_text_after_new_line,
        )
        new_summaries_per_prompt_index = 0
        for index in indices_of_degenerate_summaries:
            final_summaries_for_current_prompt = []
            final_log_probabilities_for_current_prompt = []
            all_summaries_per_prompt = (
                summaries_per_prompt[index]
                + new_summaries_per_prompt[new_summaries_per_prompt_index]
            )
            if number_of_log_probabilities > 0:
                assert log_probabilities_per_prompt is not None
                assert new_log_probabilities_per_prompt is not None
                all_log_probabilities_per_prompt = (
                    log_probabilities_per_prompt[index]
                    + new_log_probabilities_per_prompt[new_summaries_per_prompt_index]
                )
                assert len(all_summaries_per_prompt) == len(
                    all_log_probabilities_per_prompt
                )
            final_summary_per_prompt_indices = []
            for i, summary in enumerate(all_summaries_per_prompt):
                if (
                    summary not in final_summaries_for_current_prompt
                    and not texts_have_forbidden_substrings([summary], forbidden_texts)
                ):
                    final_summaries_for_current_prompt.append(summary)
                    final_summary_per_prompt_indices.append(i)
                    if number_of_log_probabilities > 0:
                        final_log_probabilities_for_current_prompt.append(
                            all_log_probabilities_per_prompt[i]
                        )
            # if our budget wasn't enough to create valid summaries (i.e. unique and with no forbiddent text) we just add the remaining ones according to order of generation
            j = 0
            while (
                len(final_summaries_for_current_prompt)
                < number_of_generations_per_prompt
            ):
                # in the first round we try to add duplicates that at least don't have a forbidden word
                if (
                    j not in final_summary_per_prompt_indices
                    and not texts_have_forbidden_substrings(
                        [all_summaries_per_prompt[j]], forbidden_texts
                    )
                ):
                    final_summaries_for_current_prompt.append(
                        all_summaries_per_prompt[j]
                    )
                    final_summary_per_prompt_indices.append(j)
                    if number_of_log_probabilities > 0:
                        final_log_probabilities_for_current_prompt.append(
                            all_log_probabilities_per_prompt[j]
                        )
                j += 1
                if j == len(all_summaries_per_prompt):
                    break

            # in the second round we just add all remaining, since our budget wasnt enough to create smaples without forbidden text
            j = 0
            while (
                len(final_summaries_for_current_prompt)
                < number_of_generations_per_prompt
            ):
                if j not in final_summary_per_prompt_indices:
                    final_summaries_for_current_prompt.append(
                        all_summaries_per_prompt[j]
                    )
                    final_summary_per_prompt_indices.append(j)
                    if number_of_log_probabilities > 0:
                        final_log_probabilities_for_current_prompt.append(
                            all_log_probabilities_per_prompt[j]
                        )
                j += 1

            summaries_per_prompt[index] = final_summaries_for_current_prompt
            if number_of_log_probabilities > 0:
                assert log_probabilities_per_prompt is not None
                log_probabilities_per_prompt[
                    index
                ] = final_log_probabilities_for_current_prompt
            new_summaries_per_prompt_index += 1

    return summaries_per_prompt, log_probabilities_per_prompt


def get_preferred_summary(comparison_preference: str) -> str:
    if "A" in comparison_preference:
        assert "B" not in comparison_preference
        return " A"
    elif "B" in comparison_preference:
        assert "A" not in comparison_preference
        return " B"
    else:
        raise ValueError(
            "Invalid comparison preference: {}".format(comparison_preference)
        )


def generate_feedback_refinement_prompt_and_completion(
    title: str, post: str, summary: str, feedback: str, refinement: str
) -> Tuple[str, str]:
    refinement_instructions = "Write an excellent summary that incorporates the feedback on the given summary and is better than the given summary.\n\n"
    prompt = refinement_instructions + _title_post_format(title, post)
    prompt += "Summary: " + summary + "\n\n"
    prompt += "Feedback on summary:"

    feedback = re.sub(" +", " ", feedback)
    if feedback[-1] == " ":
        feedback = feedback[:-1]
    refinement = re.sub(" +", " ", refinement)
    if refinement[-1] == " ":
        refinement = refinement[:-1]

    # note at inference time we need to generate feedback + refinement so we can't just limit the output to 48 tokens.
    # thus we need to add a special end token
    completion = " " + feedback + "\n\n" + "Improved TL;DR: " + refinement + "\n###"
    return prompt, completion
