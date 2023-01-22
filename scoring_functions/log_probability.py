import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from gpt3_language_model import GPT3LanguageModel
from scoring_functions.scoring_function_base import BaseScoringFunction
from utilities.summarization_utilities import (
    feedback_given_refinement_prompt_format,
    refinement_given_summary_feedback_prompt_format,
)


class LogProbabilityScoring(BaseScoringFunction):
    def __init__(
        self,
        data: pd.DataFrame,
        number_of_refinements: int,
        scoring_function_name: str,
        scoring_probabilitiy_methods: List[str],
    ) -> None:
        super(LogProbabilityScoring, self).__init__(
            data=data,
            number_of_refinements=number_of_refinements,
            scoring_function_name=scoring_function_name,
        )
        self.scoring_probabilitiy_methods = scoring_probabilitiy_methods
        self.refinement_given_summary_feedback_instruction = "Write an excellent summary that incorporates the feedback on the given summary and is better than the given summary.\n\n"
        self.feedback_given_refinement_instruction = "Write feedback that critiques the given summary. The feedback should help write an improved and excellent summary.\n\n"

    def select_refinement(
        self, model_name: str, feedback_summary_name: str
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        refinement_probability_scores = self.calculate_refinement_scores(
            model_name, feedback_summary_name
        )
        (
            refinement_ranking,
            refinement_ranking_scores,
        ) = self.calculate_ranking_given_refinement_scores(
            refinement_probability_scores
        )

        # Using only a few out of all probability methods to make selection.
        selected_refinements_per_method: Dict[str, List[str]] = {}
        ordered_refinement_selection_per_method: Dict[str, List[List[str]]] = {}
        for scoring_method in self.scoring_probabilitiy_methods:
            selected_refinements_per_method[
                scoring_method.replace("ranking", "selected_refinement")
            ] = []
            ordered_refinement_selection_per_method[
                scoring_method.replace("ranking", "ordered_refinement_selection")
            ] = []

        for sample_index in range(self.number_of_samples):
            for scoring_method in self.scoring_probabilitiy_methods:
                (
                    selected_refinement,
                    refinement_selection_letters,
                ) = self._select_best_refinement_for_scoring_method(
                    refinement_ranking, scoring_method, sample_index
                )
                selected_refinements_per_method[
                    scoring_method.replace("ranking", "selected_refinement")
                ].append(selected_refinement)
                ordered_refinement_selection_per_method[
                    scoring_method.replace("ranking", "ordered_refinement_selection")
                ].append(refinement_selection_letters)

        return (
            pd.concat(
                [
                    pd.DataFrame(selected_refinements_per_method),
                    pd.DataFrame(ordered_refinement_selection_per_method),
                ],
                axis=1,
            ),
            pd.concat(
                [refinement_ranking_scores, refinement_probability_scores], axis=1
            ),
        )

    def _select_best_refinement_for_scoring_method(
        self, refinement_ranking: pd.DataFrame, scoring_method: str, sample_index: int
    ) -> Tuple[str, List[str]]:
        selected_refinement_index = refinement_ranking.iloc[sample_index][
            scoring_method
        ][0]
        refinement_selection_letter = self.alphabet[selected_refinement_index]
        selected_refinement = self.data.iloc[sample_index][
            "generated_refinement_{}".format(refinement_selection_letter)
        ]
        refinement_selection_letters = [
            self.alphabet[refinement_index]
            for refinement_index in refinement_ranking.iloc[sample_index][
                scoring_method
            ]
        ]
        return selected_refinement, refinement_selection_letters

    def calculate_ranking_given_refinement_scores(
        self, refinement_scores: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        feedback_given_text_refinements_average_probabilities_across_refinements = []
        for sample_index in range(self.number_of_samples):
            assert (
                len(
                    refinement_scores[
                        "probability_feedback_given_text_refinement"
                    ].iloc[sample_index]
                )
                == self.number_of_refinements
            )
            feedback_given_text_refinements_average_probabilities_across_refinements.append(
                np.mean(
                    refinement_scores[
                        "probability_feedback_given_text_refinement"
                    ].iloc[sample_index]
                )
            )

        expected_feedback_probability = np.mean(
            feedback_given_text_refinements_average_probabilities_across_refinements
        )

        # sorted items
        all_rankings_r_given_xyf = []
        all_rankings_r_given_x = []
        all_rankings_f_given_xr = []
        all_rankings_f_given_xr_minus_expected_feedback = []
        all_rankings_f_given_xr_minus_r_given_xyf = []
        all_rankings_f_given_xr_minus_r_given_x = []
        all_rankings_f_given_xr_minus_r_given_xyf_minus_expected_feedback = []
        all_rankings_f_given_xr_minus_r_given_x_minus_expected_feedback = []
        all_rankings_joint_probability_xyfr = []
        all_rankings_joint_probability_xrf = []

        # for some scoring functions we calculate specific scores. We thus have a scores array for each score.
        all_ranking_scores_r_given_xyf = []
        all_ranking_scores_r_given_x = []
        all_ranking_scores_f_given_xr = []
        all_ranking_scores_f_given_xr_minus_expected_feedback = []
        all_ranking_scores_f_given_xr_minus_r_given_xyf = []
        all_ranking_scores_f_given_xr_minus_r_given_x = []
        all_ranking_scores_f_given_xr_minus_r_given_xyf_minus_expected_feedback = []
        all_ranking_scores_f_given_xr_minus_r_given_x_minus_expected_feedback = []
        all_ranking_scores_joint_probability_xyfr = []
        all_ranking_scores_joint_probability_xrf = []

        for sample_index in range(self.number_of_samples):
            sample_refinement_scores = refinement_scores.iloc[sample_index]

            sample_probability_refinement_given_text_summary_feedback = np.array(
                sample_refinement_scores[
                    "probability_refinement_given_text_summary_feedback"
                ]
            )

            sample_probability_refinement_given_text = np.array(
                sample_refinement_scores["probability_refinement_given_text"]
            )
            sample_probability_feedback_given_text_refinement = np.array(
                sample_refinement_scores["probability_feedback_given_text_refinement"]
            )

            sample_joint_probability_text_summary_feedback_refinement = np.array(
                sample_refinement_scores[
                    "joint_probability_text_summary_feedback_refinement"
                ]
            )
            sample_joint_probability_text_refinement_feedback = np.array(
                sample_refinement_scores["joint_probability_text_refinement_feedback"]
            )

            # argsort ranks according to smallest first. Thus to maximize use negative vector, and minimize use given vector
            # Also, to make names shorter we use the following abbreviations:
            # text = x, summary = y, feedback = f, refinement = r

            # maximize
            ranking_r_given_xyf = np.argsort(
                -sample_probability_refinement_given_text_summary_feedback
            )
            ranking_score_r_given_xyf = (
                sample_probability_refinement_given_text_summary_feedback
            )

            # maximize
            ranking_r_given_x = np.argsort(-sample_probability_refinement_given_text)
            ranking_score_r_given_x = sample_probability_refinement_given_text

            # minimize
            ranking_f_given_xr = np.argsort(
                sample_probability_feedback_given_text_refinement
            )
            ranking_score_f_given_xr = sample_probability_feedback_given_text_refinement

            # closest to 0 is best
            ranking_score_f_given_xr_minus_expected_feedback = np.abs(
                np.subtract(
                    sample_probability_feedback_given_text_refinement,
                    expected_feedback_probability,
                )
            )
            ranking_f_given_xr_minus_expected_feedback = np.argsort(
                ranking_score_f_given_xr_minus_expected_feedback
            )

            # minimize
            ranking_score_f_given_xr_minus_r_given_xyf = (
                sample_probability_feedback_given_text_refinement
                - sample_probability_refinement_given_text_summary_feedback
            )
            ranking_f_given_xr_minus_r_given_xyf = np.argsort(
                ranking_score_f_given_xr_minus_r_given_xyf
            )

            # minimize
            ranking_score_f_given_xr_minus_r_given_x = (
                sample_probability_feedback_given_text_refinement
                - sample_probability_refinement_given_text
            )
            ranking_f_given_xr_minus_r_given_x = np.argsort(
                ranking_score_f_given_xr_minus_r_given_x
            )

            # closest to 0 is best
            ranking_score_f_given_xr_minus_r_given_xyf_minus_expected_feedback = np.abs(
                np.subtract(
                    sample_probability_feedback_given_text_refinement
                    - sample_probability_refinement_given_text_summary_feedback,
                    expected_feedback_probability,
                )
            )
            ranking_f_given_xr_minus_r_given_xyf_minus_expected_feedback = np.argsort(
                ranking_score_f_given_xr_minus_r_given_xyf_minus_expected_feedback
            )

            # closest to 0 is best
            ranking_score_f_given_xr_minus_r_given_x_minus_expected_feedback = np.abs(
                np.subtract(
                    sample_probability_feedback_given_text_refinement
                    - sample_probability_refinement_given_text,
                    expected_feedback_probability,
                )
            )
            ranking_f_given_xr_minus_r_given_x_minus_expected_feedback = np.argsort(
                ranking_score_f_given_xr_minus_r_given_x_minus_expected_feedback
            )

            # maximize
            ranking_joint_probability_xyfr = np.argsort(
                -sample_joint_probability_text_summary_feedback_refinement
            )
            ranking_score_joint_probability_xyfr = (
                sample_joint_probability_text_summary_feedback_refinement
            )

            ranking_joint_probability_xrf = np.argsort(
                sample_joint_probability_text_refinement_feedback
            )
            ranking_score_joint_probability_xrf = (
                sample_joint_probability_text_refinement_feedback
            )

            all_rankings_r_given_xyf.append(ranking_r_given_xyf)
            all_rankings_r_given_x.append(ranking_r_given_x)
            all_rankings_f_given_xr.append(ranking_f_given_xr)
            all_rankings_f_given_xr_minus_expected_feedback.append(
                ranking_f_given_xr_minus_expected_feedback
            )
            all_rankings_f_given_xr_minus_r_given_xyf.append(
                ranking_f_given_xr_minus_r_given_xyf
            )
            all_rankings_f_given_xr_minus_r_given_x.append(
                ranking_f_given_xr_minus_r_given_x
            )
            all_rankings_f_given_xr_minus_r_given_xyf_minus_expected_feedback.append(
                ranking_f_given_xr_minus_r_given_xyf_minus_expected_feedback
            )
            all_rankings_f_given_xr_minus_r_given_x_minus_expected_feedback.append(
                ranking_f_given_xr_minus_r_given_x_minus_expected_feedback
            )
            all_rankings_joint_probability_xyfr.append(ranking_joint_probability_xyfr)
            all_rankings_joint_probability_xrf.append(ranking_joint_probability_xrf)

            all_ranking_scores_r_given_xyf.append(ranking_score_r_given_xyf)
            all_ranking_scores_r_given_x.append(ranking_score_r_given_x)
            all_ranking_scores_f_given_xr.append(ranking_score_f_given_xr)
            all_ranking_scores_f_given_xr_minus_expected_feedback.append(
                ranking_score_f_given_xr_minus_expected_feedback
            )
            all_ranking_scores_f_given_xr_minus_r_given_xyf.append(
                ranking_score_f_given_xr_minus_r_given_xyf
            )
            all_ranking_scores_f_given_xr_minus_r_given_x.append(
                ranking_score_f_given_xr_minus_r_given_x
            )
            all_ranking_scores_f_given_xr_minus_r_given_xyf_minus_expected_feedback.append(
                ranking_score_f_given_xr_minus_r_given_xyf_minus_expected_feedback
            )
            all_ranking_scores_f_given_xr_minus_r_given_x_minus_expected_feedback.append(
                ranking_score_f_given_xr_minus_r_given_x_minus_expected_feedback
            )
            all_ranking_scores_joint_probability_xyfr.append(
                ranking_score_joint_probability_xyfr
            )
            all_ranking_scores_joint_probability_xrf.append(
                ranking_score_joint_probability_xrf
            )

        all_probability_rankings = pd.DataFrame(
            {
                "ranking_r_given_xyf": all_rankings_r_given_xyf,
                "ranking_r_given_x": all_rankings_r_given_x,
                "ranking_f_given_xr": all_rankings_f_given_xr,
                "ranking_f_given_xr_minus_expected_feedback": all_rankings_f_given_xr_minus_expected_feedback,
                "ranking_f_given_xr_minus_r_given_xyf": all_rankings_f_given_xr_minus_r_given_xyf,
                "ranking_f_given_xr_minus_r_given_x": all_rankings_f_given_xr_minus_r_given_x,
                "ranking_f_given_xr_minus_r_given_xyf_minus_expected_feedback": all_rankings_f_given_xr_minus_r_given_xyf_minus_expected_feedback,
                "ranking_f_given_xr_minus_r_given_x_minus_expected_feedback": all_rankings_f_given_xr_minus_r_given_x_minus_expected_feedback,
                "ranking_joint_probability_xyfr": all_rankings_joint_probability_xyfr,
                "ranking_joint_probability_xrf": all_rankings_joint_probability_xrf,
            }
        )
        all_ranking_scores = pd.DataFrame(
            {
                "ranking_scores_r_given_xyf": all_ranking_scores_r_given_xyf,
                "ranking_scores_r_given_x": all_ranking_scores_r_given_x,
                "ranking_scores_f_given_xr": all_ranking_scores_f_given_xr,
                "ranking_scores_f_given_xr_minus_expected_feedback": all_ranking_scores_f_given_xr_minus_expected_feedback,
                "ranking_scores_f_given_xr_minus_r_given_xyf": all_ranking_scores_f_given_xr_minus_r_given_xyf,
                "ranking_scores_f_given_xr_minus_r_given_x": all_ranking_scores_f_given_xr_minus_r_given_x,
                "ranking_scores_f_given_xr_minus_r_given_xyf_minus_expected_feedback": all_ranking_scores_f_given_xr_minus_r_given_xyf_minus_expected_feedback,
                "ranking_scores_f_given_xr_minus_r_given_x_minus_expected_feedback": all_ranking_scores_f_given_xr_minus_r_given_x_minus_expected_feedback,
                "ranking_scores_joint_probability_xyfr": all_ranking_scores_joint_probability_xyfr,
                "ranking_scores_joint_probability_xrf": all_ranking_scores_joint_probability_xrf,
            }
        )

        return all_probability_rankings, all_ranking_scores

    def calculate_refinement_scores(
        self, model_name: str, feedback_summary_name: str
    ) -> pd.DataFrame:
        model = GPT3LanguageModel(model_name=model_name)

        refinement_given_summary_feedback_completions = []
        feedback_given_refinement_completions = []
        for sample_index in tqdm(range(self.number_of_samples)):
            sample = self.data.iloc[sample_index]

            refinement_given_summary_feedback_prompts_per_sample = []
            feedback_given_refinement_prompts_per_sample = []
            for refinement_index in range(self.number_of_refinements):
                refinement_selection_letter = self.alphabet[refinement_index]
                curent_refinement = self.data.iloc[sample_index][
                    "generated_refinement_{}".format(refinement_selection_letter)
                ]

                refinement_given_summary_feedback_prompt = (
                    self.refinement_given_summary_feedback_instruction
                    + refinement_given_summary_feedback_prompt_format(
                        title=sample["title"],
                        post=sample["post"],
                        summary=sample[feedback_summary_name],
                        feedback=sample["feedback"],
                        refinement=curent_refinement,
                    )
                )
                refinement_given_summary_feedback_prompts_per_sample.append(
                    refinement_given_summary_feedback_prompt
                )

                feedback_given_refinement_prompt = (
                    self.feedback_given_refinement_instruction
                    + feedback_given_refinement_prompt_format(
                        title=sample["title"],
                        post=sample["post"],
                        refinement=curent_refinement,
                        feedback=sample["feedback"],
                    )
                )
                feedback_given_refinement_prompts_per_sample.append(
                    feedback_given_refinement_prompt
                )

            refinement_given_summary_feedback_completions_per_sample = model.generate_completion(
                prompts=refinement_given_summary_feedback_prompts_per_sample,
                batch_size=1,
                temperature=0.0,
                number_of_log_probabilities=1,
                max_tokens_to_generate=0,
                echo=True,
                top_p=0,
            )

            feedback_given_refinement_completion_per_sample = model.generate_completion(
                prompts=feedback_given_refinement_prompts_per_sample,
                batch_size=1,
                temperature=0.0,
                number_of_log_probabilities=1,
                max_tokens_to_generate=0,
                echo=True,
                top_p=0,
            )

            assert isinstance(
                refinement_given_summary_feedback_completions_per_sample, List
            )
            assert isinstance(feedback_given_refinement_completion_per_sample, List)
            assert len(refinement_given_summary_feedback_completions_per_sample) == len(
                feedback_given_refinement_completion_per_sample
            )
            for i in range(
                len(refinement_given_summary_feedback_completions_per_sample)
            ):
                assert isinstance(
                    refinement_given_summary_feedback_completions_per_sample[i], dict
                )
                assert isinstance(
                    feedback_given_refinement_completion_per_sample[i], dict
                )

            refinement_given_summary_feedback_completions.append(
                refinement_given_summary_feedback_completions_per_sample
            )
            feedback_given_refinement_completions.append(
                feedback_given_refinement_completion_per_sample
            )

        probability_refinement_given_text_summary_feedback = self.get_probability_refinement_given_text_summary_feedback(
            refinement_given_summary_feedback_completions  # type: ignore
        )
        probability_refinement_given_text = self.get_probability_refinement_given_text(
            feedback_given_refinement_completions  # type: ignore
        )
        probability_feedback_given_text_refinement = self.get_probability_feedback_text_refinement(
            feedback_given_refinement_completions  # type: ignore
        )
        joint_probability_text_summary_feedback_refinement = self.get_joint_probability(
            refinement_given_summary_feedback_completions,  # type: ignore
        )
        joint_probability_text_refinement_feedback = self.get_joint_probability(
            feedback_given_refinement_completions,  # type: ignore
        )
        probability_scores = pd.DataFrame(
            {
                "probability_refinement_given_text_summary_feedback": probability_refinement_given_text_summary_feedback,
                "probability_refinement_given_text": probability_refinement_given_text,
                "probability_feedback_given_text_refinement": probability_feedback_given_text_refinement,
                "joint_probability_text_summary_feedback_refinement": joint_probability_text_summary_feedback_refinement,
                "joint_probability_text_refinement_feedback": joint_probability_text_refinement_feedback,
            }
        )
        return probability_scores

    def get_probability_refinement_given_text_summary_feedback(
        self, completions: List[List[Dict[str, Any]]]
    ) -> List[List[np.ndarray]]:
        """p_r_xyf"""
        probability_refinement_given_text_summary_feedback = []
        for sample_completion in completions:
            p_r_xyf_per_sample = []
            assert len(sample_completion) == self.number_of_refinements
            for refinement_index in range(len(sample_completion)):
                refinement_completion = sample_completion[refinement_index]
                self._assert_prompt_is_refinement_given_summary_feedback(
                    refinement_completion["text"],
                    refinement_completion["log_probabilities"]["text_offset"],
                )
                refinement = re.search(
                    "Improved TL;DR:(.+)\n\n", refinement_completion["text"]
                )
                assert refinement is not None
                refinement = refinement.group(1)
                # use rfind since refinement could be duplicate of original summary.
                refinement_start_index = refinement_completion["text"].rfind(refinement)
                probability_start_index = refinement_completion["log_probabilities"][
                    "text_offset"
                ].index(refinement_start_index)
                # rfind returns rightmost index
                prompt_end_index = refinement_completion["text"].rfind("\n\n")
                assert prompt_end_index != -1, "Failed to get last newlines"
                probability_end_index = refinement_completion["log_probabilities"][
                    "text_offset"
                ].index(prompt_end_index)
                assert (
                    refinement_completion["log_probabilities"]["tokens"][
                        probability_end_index
                    ]
                    == "\n\n"
                )
                # take everything up to but excluding \n\n
                p_r_xyf_per_sample.append(
                    np.mean(
                        refinement_completion["log_probabilities"]["token_logprobs"][
                            probability_start_index:probability_end_index
                        ]
                    )
                )
            probability_refinement_given_text_summary_feedback.append(
                p_r_xyf_per_sample
            )
        return probability_refinement_given_text_summary_feedback

    def get_probability_feedback_text_refinement(
        self, completions: List[List[Dict[str, Any]]]
    ) -> List[List[np.ndarray]]:
        """p_f_xr"""
        probability_feedback_given_summary_refinement = []
        for sample_completion in completions:
            p_f_xr_per_sample = []
            for feedback_index in range(len(sample_completion)):
                feedback_completion = sample_completion[feedback_index]
                self._assert_prompt_is_feedback_given_refinement(
                    feedback_completion["text"],
                    feedback_completion["log_probabilities"]["text_offset"],
                )

                feedback = re.search(
                    "Feedback on summary:(.+)\n\n", feedback_completion["text"]
                )
                assert feedback is not None
                feedback = feedback.group(1)
                feedack_start_index = feedback_completion["text"].index(feedback)
                probability_start_index = feedback_completion["log_probabilities"][
                    "text_offset"
                ].index(feedack_start_index)
                prompt_end_index = feedback_completion["text"].rfind("\n\n")
                assert prompt_end_index != -1, "Failed to get last newlines"
                probability_end_index = feedback_completion["log_probabilities"][
                    "text_offset"
                ].index(prompt_end_index)
                p_f_xr_per_sample.append(
                    np.mean(
                        feedback_completion["log_probabilities"]["token_logprobs"][
                            probability_start_index:probability_end_index
                        ]
                    )
                )
            probability_feedback_given_summary_refinement.append(p_f_xr_per_sample)
        return probability_feedback_given_summary_refinement

    def get_probability_refinement_given_text(
        self, completions: List[List[Dict[str, Any]]]
    ) -> List[List[np.ndarray]]:
        """p_r_x"""
        probability_refinement_given_text = []
        for sample_completion in completions:
            p_r_x_per_sample = []
            for feedback_index in range(len(sample_completion)):
                feedback_completion = sample_completion[feedback_index]
                self._assert_prompt_is_feedback_given_refinement(
                    feedback_completion["text"],
                    feedback_completion["log_probabilities"]["text_offset"],
                )

                refinement = re.search(
                    "TL;DR:(.+)\n\nFeedback on summary:", feedback_completion["text"]
                )
                assert refinement is not None
                refinement = refinement.group(1)
                refinement_start_index = feedback_completion["text"].index(refinement)
                probability_start_index = feedback_completion["log_probabilities"][
                    "text_offset"
                ].index(refinement_start_index)
                feedback_end_index = feedback_completion["text"].index(
                    "\n\nFeedback on summary:"
                )
                probability_end_index = feedback_completion["log_probabilities"][
                    "text_offset"
                ].index(feedback_end_index)
                p_r_x_per_sample.append(
                    np.mean(
                        feedback_completion["log_probabilities"]["token_logprobs"][
                            probability_start_index:probability_end_index
                        ]
                    )
                )
            probability_refinement_given_text.append(p_r_x_per_sample)
        return probability_refinement_given_text

    def get_joint_probability(
        self, completions: List[List[Dict[str, Any]]],
    ) -> List[List[np.ndarray]]:
        joint_probabilities = []
        for sample_completion in completions:
            joint_probabilities_per_sample = []
            for index in range(len(sample_completion)):
                assert (
                    "Feedback on summary" in sample_completion[index]["text"]
                ), "The completions do not contain feedback"
                # Avoid initial None and final \n\n
                joint_probabilities_per_sample.append(
                    np.mean(
                        sample_completion[index]["log_probabilities"]["token_logprobs"][
                            1:-1
                        ]
                    )
                )

            joint_probabilities.append(joint_probabilities_per_sample)
        return joint_probabilities

    def _assert_prompt_is_refinement_given_summary_feedback(
        self, prompt: str, text_offset: List[int]
    ) -> None:
        assert "Title" in prompt
        assert "Text" in prompt
        assert "Summary" in prompt
        assert "Feedback on summary" in prompt
        assert "Improved TL;DR" in prompt

        title = re.search("Title:(.+)\n\n", prompt)
        assert title is not None
        title = title.group(1)  # type: ignore
        text = re.search("Text:(.+)\n\n", prompt)
        assert text is not None
        text = text.group(1)  # type: ignore
        assert isinstance(text, str)

        summary = re.search("Summary:(.+)\n\n", prompt)
        assert summary is not None
        summary = summary.group(1)  # type: ignore
        assert isinstance(summary, str)

        feedback = re.search("Feedback on summary:(.+)\n\n", prompt)
        assert feedback is not None
        feedback = feedback.group(1)  # type: ignore
        assert isinstance(feedback, str)

        refinement = re.search("Improved TL;DR:(.+)\n\n", prompt)
        assert refinement is not None
        refinement = refinement.group(1)  # type: ignore
        assert isinstance(refinement, str)

        token_index_of_title = self.get_token_index_of_sentence(
            prompt, title, text_offset
        )
        token_index_of_text = self.get_token_index_of_sentence(
            prompt, text, text_offset
        )
        token_index_of_summary = self.get_token_index_of_sentence(
            prompt, summary, text_offset
        )
        token_index_of_feedback = self.get_token_index_of_sentence(
            prompt, feedback, text_offset
        )
        refinement_start = prompt.find("Improved TL;DR:")
        token_index_of_refinement = self.get_token_index_of_sentence(
            prompt, refinement, text_offset, refinement_start
        )
        if (
            token_index_of_title
            < token_index_of_text
            < token_index_of_summary
            < token_index_of_feedback
            < token_index_of_refinement
        ):
            assert (
                token_index_of_title
                < token_index_of_text
                < token_index_of_summary
                < token_index_of_feedback
                < token_index_of_refinement
            ), "Order of Prompt is incorrect."
        else:
            print("Order of Prompt is incorrect.")
            print(prompt)
            print(
                "{} < {} < {} < {} < {}".format(
                    token_index_of_title,
                    token_index_of_text,
                    token_index_of_summary,
                    token_index_of_feedback,
                    token_index_of_refinement,
                )
            )

    def _assert_prompt_is_feedback_given_refinement(
        self, prompt: str, text_offset: List[int]
    ) -> None:
        assert "Title" in prompt
        assert "Text" in prompt
        if "Summary" in prompt:
            print(prompt)
        assert "Feedback on summary:" in prompt
        assert "TL;DR" in prompt

        title = re.search("Title:(.+)\n\n", prompt)
        assert title is not None
        title = title.group(1)  # type: ignore
        assert isinstance(title, str)

        text = re.search("Text:(.+)\n\n", prompt)
        assert text is not None
        text = text.group(1)
        assert isinstance(text, str)

        refinement = re.search("TL;DR:(.+)\n\n", prompt)
        assert refinement is not None
        refinement = refinement.group(1)
        assert isinstance(refinement, str)

        feedback = re.search("Feedback on summary:(.+)\n\n", prompt)
        assert feedback is not None
        feedback = feedback.group(1)
        assert isinstance(feedback, str)

        token_index_of_title = self.get_token_index_of_sentence(
            prompt, title, text_offset
        )
        token_index_of_text = self.get_token_index_of_sentence(
            prompt, text, text_offset
        )
        token_index_of_feedback = self.get_token_index_of_sentence(
            prompt, feedback, text_offset
        )
        token_index_of_refinement = self.get_token_index_of_sentence(
            prompt, refinement, text_offset
        )
        if (
            token_index_of_title
            < token_index_of_text
            < token_index_of_refinement
            < token_index_of_feedback
        ):
            assert (
                token_index_of_title
                < token_index_of_text
                < token_index_of_refinement
                < token_index_of_feedback
            ), "Order of Prompt is incorrect."
        else:
            print("Order of Prompt is incorrect.")
            print(prompt)
            print(
                "{} < {} < {} < {}".format(
                    token_index_of_title,
                    token_index_of_text,
                    token_index_of_refinement,
                    token_index_of_feedback,
                )
            )

    def get_token_index_of_sentence(
        self, text: str, sentence: str, text_offset: List[int], start_index: int = None
    ) -> int:
        if start_index:
            sentence_index = text.find(sentence, start_index)
        else:
            sentence_index = text.find(sentence)
        return text_offset.index(sentence_index)
