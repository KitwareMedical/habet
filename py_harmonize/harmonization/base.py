from abc import ABC, abstractmethod

class HarmonizationMethod(ABC):
    def __init__(self, data_matrix, df, site_colname, covariate_cols=None):
        self.data_matrix = data_matrix
        self.df = df
        self.site_colname = site_colname
        self.covariate_cols = covariate_cols

    @abstractmethod
    def _harmonize(self):
        pass

    def harmonize(self):
        return self._harmonize()