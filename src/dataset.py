import abc
import enum
import uuid
import typing as t
import numpy as np
import pandas as pd


class Columns(str, enum.Enum):
    ID = "id"
    TIMESTAMP = "timestamp"
    LABEL = "label"


class BaseDataset:

    def __init__(self, sample_size: int, feature_size: int, missing_ratio: float, seed: int | None = None):
        self.sample_size = sample_size
        self.feature_size = feature_size
        self.missing_ratio = missing_ratio
        if seed is not None:
            np.random.seed(seed)

    @abc.abstractmethod
    def generate(self) -> pd.DataFrame:
        """
        Generate pandas dataframe.
        """
        raise NotImplementedError

    def _generate_ids(self, sample_size: int) -> list[str]:
        return [str(uuid.uuid4()) for _ in range(sample_size)]


class TimeSeriesDataset(BaseDataset):
    """
    Engagement score dataset.
    """

    key_column = Columns.ID
    timestamp_column = Columns.TIMESTAMP

    def __init__(
        self, sample_size: int, feature_size: int, start_date: str, end_date: str, missing_ratio: float, seed: int = 42
    ):
        super().__init__(sample_size, feature_size, missing_ratio, seed)
        self.start_date = start_date
        self.end_date = end_date

    def generate(self) -> pd.DataFrame:
        """
        Generate time series dataset.
        The generated data type only contain numerical data for each column exclude key column.
        especially, the timestamp column is date type.
        """
        ids = self._generate_ids(self.sample_size)
        date_range = pd.date_range(start=self.start_date, end=self.end_date)
        data: list[dict[str, t.Any]] = [
            {
                **{self.key_column.value: id_, self.timestamp_column.value: timestamp},
                **self._generate_numerical_features(self.feature_size),
            }
            for id_ in ids
            for timestamp in date_range
        ]
        return pd.DataFrame(data)

    def _generate_numerical_features(self, size: int) -> dict[str, t.Union[float, np.ndarray]]:
        return {
            f"feature_{idx}": np.where(
                np.random.rand(size) < self.missing_ratio,
                np.nan,
                np.random.randn(size),
            )
            for idx in range(1, self.feature_size + 1)
        }


class CategoryDataset(BaseDataset):
    """
    Customer profile dataset.
    """

    key_column = Columns.ID

    def __init__(self, sample_size: int, feature_size: int, missing_ratio: float, seed: int = 42):
        super().__init__(sample_size, feature_size, missing_ratio, seed)

    def generate(self) -> pd.DataFrame:
        """
        Generate category dataset.
        The generated data type only contain catagorical data for each column exclude key column.
        """
        return pd.DataFrame()
