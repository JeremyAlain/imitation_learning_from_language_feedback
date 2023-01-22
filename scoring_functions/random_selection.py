from typing import Optional, Tuple

import numpy as np
import pandas as pd

from scoring_functions.scoring_function_base import BaseScoringFunction


class RandomScoring(BaseScoringFunction):
    """Randomly select a refinement"""

    def select_refinement(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        selected_refinements = []
        ordered_refinement_selection = []
        for i in range(self.number_of_samples):
            refinement_indices = np.arange(self.number_of_refinements)
            np.random.shuffle(refinement_indices)
            random_refinement_selection_index = refinement_indices[0]
            random_refinement_selection_letter = self.alphabet[
                random_refinement_selection_index
            ]
            selected_refinement = self.data.iloc[i][
                "generated_refinement_{}".format(random_refinement_selection_letter)
            ]
            random_refinement_selection_letters = [
                self.alphabet[refinement_index]
                for refinement_index in refinement_indices
            ]
            ordered_refinement_selection.append(random_refinement_selection_letters)
            selected_refinements.append(selected_refinement)
        selected_refinements_pd = pd.DataFrame(
            {
                self.scoring_function_name
                + "_selected_refinement": selected_refinements,
                self.scoring_function_name
                + "_ordered_refinement_selection": ordered_refinement_selection,
            }
        )
        return selected_refinements_pd, None
