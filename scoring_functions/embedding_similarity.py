from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from gpt3_language_model import GPT3TextEmbeddingModel
from scoring_functions.scoring_function_base import BaseScoringFunction


class FeedbackRefinementEmbeddingSimilarityScoring(BaseScoringFunction):
    """Select the refinements by calculating the openai embedding of feedback and refinement.
    Then either take the max or min."""

    def select_refinement(
        self, model_name: str, select_argmax: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        selected_refinements = []
        ordered_refinement_selection = []
        refinement_scores = self.calculate_refinement_scores(model_name)
        for sample_index in range(self.number_of_samples):
            sample_cosine_similarities = np.squeeze(
                refinement_scores.iloc[sample_index]["cosine_similarities"]
            )
            if select_argmax:
                selected_refinement_indices = np.argsort(sample_cosine_similarities)[
                    ::-1
                ]
            else:
                selected_refinement_indices = np.argsort(sample_cosine_similarities)
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

    def calculate_refinement_scores(self, model_name: str) -> pd.DataFrame:
        feedback_embeddings = []
        refinement_embeddings = []
        cosine_similarities = []

        embedding_model = GPT3TextEmbeddingModel(model_name=model_name)
        for sample_index in tqdm(range(self.number_of_samples)):
            feedback = self.data.iloc[sample_index]["feedback"]
            feedback_embedding = embedding_model.embed_text(feedback)
            feedback_embeddings.append(feedback_embedding)

            cosine_similarities_per_sample = []
            refinement_embeddings_per_sample = []
            for refinement_index in range(self.number_of_refinements):
                refinement_selection_letter = self.alphabet[refinement_index]
                selected_refinement = self.data.iloc[sample_index][
                    "generated_refinement_{}".format(refinement_selection_letter)
                ]
                refinement_embedding = embedding_model.embed_text(selected_refinement)
                refinement_embeddings_per_sample.append(refinement_embedding)

                cos_similarity = cosine_similarity(
                    np.expand_dims(feedback_embedding, axis=0),
                    np.expand_dims(refinement_embedding, axis=0),
                )

                cosine_similarities_per_sample.append(cos_similarity)

            refinement_embeddings.append(refinement_embeddings_per_sample)
            cosine_similarities.append(cosine_similarities_per_sample)

        refinement_scores = {
            "cosine_similarities": cosine_similarities,
            "feedback_embeddings": feedback_embeddings,
            "refinement_embeddings": refinement_embeddings,
        }

        return pd.DataFrame(refinement_scores)
