from typing import Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from gpt3_language_model import GPT3LanguageModel
from scoring_functions.scoring_function_base import BaseScoringFunction


class CausalResponseScoring(BaseScoringFunction):
    """Select the refinements by prompting GPT3 which refinement includes most of the feedback and is generally good."""

    def __init__(
        self,
        data: pd.DataFrame,
        number_of_refinements: int,
        scoring_function_name: str,
        prompt_type: str,
    ) -> None:

        self.prompt_to_use = prompt_type
        assert self.prompt_to_use in [
            "prompt_1",
            "prompt_2",
            "prompt_3",
            "prompt_4",
            "prompt_5",
        ], self.prompt_to_use

        scoring_function_name = scoring_function_name + "_" + prompt_type
        super(CausalResponseScoring, self).__init__(
            data=data,
            number_of_refinements=number_of_refinements,
            scoring_function_name=scoring_function_name,
        )

    def select_refinement(
        self, model_name: str, feedback_summary_name: str,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        selected_refinements = []
        ordered_refinement_selection = []
        refinement_scores = self.calculate_refinement_scores(
            model_name, feedback_summary_name
        )
        for sample_index in range(self.number_of_samples):
            selected_refinement_indices = np.argsort(
                refinement_scores.iloc[sample_index][
                    "causal_response_positive_probability"
                ]
            )[::-1]
            refinement_selection_letter = self.alphabet[selected_refinement_indices[0]]
            selected_refinement = self.data.iloc[sample_index][
                "generated_refinement_{}".format(refinement_selection_letter)
            ]
            selected_refinements.append(selected_refinement)
            refinement_selection_letters = [
                self.alphabet[refinement_index]
                for refinement_index in selected_refinement_indices
            ]
            ordered_refinement_selection.append(refinement_selection_letters)

        selected_refinements_pd = pd.DataFrame(
            {
                self.scoring_function_name
                + "_selected_refinement": selected_refinements,
                self.scoring_function_name
                + "_ordered_refinement_selection": ordered_refinement_selection,
            }
        )
        return selected_refinements_pd, refinement_scores

    def calculate_refinement_scores(
        self, model_name: str, feedback_summary_name: str
    ) -> pd.DataFrame:
        # Logit biases
        # token_id: 1400 = " No"
        # token_id: 3363 = " Yes"

        # token_id: 6407 = " True"
        # token_id 10352 = " False"
        # see https://beta.openai.com/tokenizer

        if self.prompt_to_use == "prompt_1":
            prompt_format = "Here's a summary of a Reddit post, feedback on the summary, and a new summary. You will be asked to determine whether the new summary incorporates the feedback provided.\n\nA good summary is a short piece of text that has the essence of the original text. A good summary tries to accomplish the same purpose and conveys the same information as the original text.\n\nPost title: {title}\n\nBelow, there's the content of the post that was summarized.\n\nOriginal post: {post}\n\nOriginal summary: {original_summary}\n\nA human then provided feedback on the above summary.\n\nFeedback: {feedback}\n\nBased on this feedback, a new summary was written.\n\nNew summary: {refinement}\n\nDoes this new summary incorporate the feedback provided? Answer Yes or No.\n\nAnswer:"
            self.logit_bias = {"3363": 100, "1400": 100}
        elif self.prompt_to_use == "prompt_2":
            prompt_format = "Post title: {title}\n\nOriginal post: {post}\n\nOriginal summary: {original_summary}\n\nFeedback: {feedback}\n\nNew summary: {refinement}\n\nQuestion: Does the new summary incorporate the feedback provided? Answer Yes or No.\n\nAnswer:"
            self.logit_bias = {"3363": 100, "1400": 100}
        elif self.prompt_to_use == "prompt_3":
            prompt_format = "You will be given a Reddit post title, its content, an original summary of that post, and feedback for that summary. Then, your goal will be to determine whether the new summary improves upon the original with respect to provided feedback.\n\nPost title: {title}\n\nPost content: {post}\n\nOriginal summary: {original_summary}\n\nFeedback: {feedback}\n\nNew summary: {refinement}\n\nQuestion: Does the new summary incorporate the feedback provided? Answer True or False.\n\nAnswer:"
            self.logit_bias = {"6407": 100, "10352": 100}
        elif self.prompt_to_use == "prompt_4":
            prompt_format = "Here's a summary of a Reddit post, feedback on the summary, and a new summary. You will be asked to determine whether the new summary incorporates the feedback provided.\n\nA good summary is a short piece of text that has the essence of the original text. A good summary tries to accomplish the same purpose and conveys the same information as the original text. Remember, you will be asked to determine whether the new summary incorporates the feedback provided.\n\nPost title: {title}\n\nBelow, there's the content of the post that was summarized. \n\nOriginal post: {post}\n\nRemember, you will be asked to determine whether the new summary incorporates the feedback provided. Here's the original summary.\n\nOriginal summary: {original_summary}\n\nRemember, you will be asked to determine whether the new summary incorporates the feedback provided. A human then provided feedback on the above summary.\n\nFeedback: {feedback}\n\nBased on this feedback, a new summary was written.\n\nNew summary: {refinement}\n\nDoes this new summary incorporate the feedback provided? Answer Yes or No.\n\nAnswer:"
            self.logit_bias = {"3363": 100, "1400": 100}
        elif self.prompt_to_use == "prompt_5":
            prompt_format = "Here's a summary of a Reddit post, feedback on the summary, and a new summary. You will be asked to determine whether the new summary incorporates the feedback provided.\n\nThe feedback was:\nFeedback: {feedback}\n\nHere's the post that was summarized in the first place.\n\nPost title: {title}\n\nOriginal post: {post}\n\nRemember, you will be asked to determine whether the new summary incorporates the feedback provided. Here's the original summary.\n\nOriginal summary: {original_summary}\n\nRemember, you will be asked to determine whether the new summary incorporates the feedback provided. A human then provided feedback on the above summary. Here's the feedback again.\n\nFeedback: {feedback}\n\nBased on this feedback, a new summary was written.\n\nNew summary: {refinement}\n\nDoes this new summary incorporate the feedback provided? Answer True or False.\n\nAnswer:"
            self.logit_bias = {"6407": 100, "10352": 100}
        else:
            raise NotImplementedError()
        model = GPT3LanguageModel(model_name=model_name)

        casual_response_normalized_positive_proabilities = []
        for sample_index in tqdm(range(self.number_of_samples)):
            sample = self.data.iloc[sample_index]

            prompts_per_sample = []
            for refinement_index in range(self.number_of_refinements):
                refinement_selection_letter = self.alphabet[refinement_index]
                selected_refinement = self.data.iloc[sample_index][
                    "generated_refinement_{}".format(refinement_selection_letter)
                ]

                prompt = prompt_format.format(
                    title=sample["title"],
                    post=sample["post"],
                    original_summary=sample[feedback_summary_name],
                    feedback=sample["feedback"],
                    refinement=selected_refinement,
                )
                prompts_per_sample.append(prompt)

            completions = model.generate_completion(
                prompts=prompts_per_sample,
                batch_size=1,
                temperature=0.0,
                number_of_log_probabilities=2,
                max_tokens_to_generate=1,
                echo=False,
                top_p=0.0,
                logit_bias=self.logit_bias,
            )
            assert len(completions) == self.number_of_refinements

            normalized_positive_probabilities_per_sample = []

            for completion in completions:
                assert isinstance(completion, dict)
                if self.logit_bias == {"3363": 100, "1400": 100}:
                    positive_log_probability = completion["log_probabilities"][
                        "top_logprobs"
                    ][0][" Yes"]
                    negative_log_probability = completion["log_probabilities"][
                        "top_logprobs"
                    ][0][" No"]
                elif self.logit_bias == {"6407": 100, "10352": 100}:
                    positive_log_probability = completion["log_probabilities"][
                        "top_logprobs"
                    ][0][" True"]
                    negative_log_probability = completion["log_probabilities"][
                        "top_logprobs"
                    ][0][" False"]
                else:
                    raise NotImplementedError()
                normalized_positive_probability = np.exp(positive_log_probability) / (
                    np.exp(positive_log_probability) + np.exp(negative_log_probability)
                )
                normalized_positive_probabilities_per_sample.append(
                    normalized_positive_probability
                )

            casual_response_normalized_positive_proabilities.append(
                normalized_positive_probabilities_per_sample
            )

        return pd.DataFrame(
            {
                "causal_response_positive_probability": casual_response_normalized_positive_proabilities
            }
        )
