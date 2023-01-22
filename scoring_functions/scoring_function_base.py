from abc import ABC, abstractmethod
import string
from typing import Optional, Tuple

import pandas as pd


class BaseScoringFunction(ABC):
    """
    A base class definining the template of Scoring Functions
    """

    def __init__(
        self, data: pd.DataFrame, number_of_refinements: int, scoring_function_name: str
    ) -> None:
        self.data = data
        self.number_of_refinements = number_of_refinements
        self.scoring_function_name = scoring_function_name

        self.number_of_samples = self.data.shape[0]
        self.alphabet = string.ascii_uppercase
        for i in range(number_of_refinements):
            assert (
                "generated_refinement_{}".format(self.alphabet[i]) in self.data.columns
            )

    @abstractmethod
    def select_refinement(
        self, *args: str
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        pass

    def calculate_refinement_scores(self, *args: str) -> pd.DataFrame:
        pass
