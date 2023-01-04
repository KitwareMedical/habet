from abc import ABC, abstractmethod
from ..registry import register_harmonization_method

class HarmonizationMethod(ABC):
    def __init__(self, data_matrix, df, site_colname, covariate_cols=None):
        self.data_matrix = data_matrix
        self.df = df
        self.site_colname = site_colname
        self.covariate_cols = covariate_cols

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_harmonization_method(cls)

    @abstractmethod
    def _harmonize(self):
        pass

    def harmonize(self):
        return self._harmonize()