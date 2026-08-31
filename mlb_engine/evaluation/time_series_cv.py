"""
Stage 5: Evaluation - Time-Series Cross-Validation
Implements strict chronological expanding and rolling window splits to prevent lookahead bias.
"""

import pandas as pd
import numpy as np
from typing import Generator, List, Tuple

class TimeSeriesMLBBacktester:
    """Chronological expanding/rolling window time-series splitter for MLB datasets."""

    def __init__(self, date_column: str = "date", min_train_games: int = 500, test_batch_size: int = 100):
        self.date_column = date_column
        self.min_train_games = min_train_games
        self.test_batch_size = test_batch_size

    def split_chronological_seasons(
        self,
        df: pd.DataFrame,
        season_column: str = "season",
        train_seasons: List[int] = [2021, 2022, 2023, 2024],
        test_season: int = 2025
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits data by full historical seasons for final out-of-sample backtesting.
        e.g. Train on 2021-2024, Test strictly on 2025.
        """
        train_df = df[df[season_column].isin(train_seasons)].copy()
        test_df = df[df[season_column] == test_season].copy()
        
        # Sort chronologically
        train_df.sort_values(by=self.date_column, inplace=True)
        test_df.sort_values(by=self.date_column, inplace=True)
        
        return train_df, test_df

    def expanding_window_generator(
        self,
        df: pd.DataFrame
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """
        Generates expanding training windows with rolling test batches without lookahead bias.
        """
        df_sorted = df.sort_values(by=self.date_column).reset_index(drop=True)
        n_samples = len(df_sorted)

        start_idx = self.min_train_games
        while start_idx < n_samples:
            end_test = min(start_idx + self.test_batch_size, n_samples)
            train_subset = df_sorted.iloc[:start_idx]
            test_subset = df_sorted.iloc[start_idx:end_test]

            yield train_subset, test_subset

            start_idx = end_test
