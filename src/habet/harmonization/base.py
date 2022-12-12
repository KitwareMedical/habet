from abc import ABC, abstractmethod

_registry = dict()

class HarmonizationMethod(ABC):
    def __init__(self, data_matrix, df, site_colname, covariate_cols=None):
        self.data_matrix = data_matrix
        self.df = df
        self.site_colname = site_colname
        self.covariate_cols = covariate_cols

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _registry[cls.__name__] = cls

    @abstractmethod
    def _harmonize(self):
        pass

    def harmonize(self):
        return self._harmonize()