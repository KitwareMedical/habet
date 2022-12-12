import warnings

from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype
from neuroCombat import neuroCombat
from .base import HarmonizationMethod

class Combat(HarmonizationMethod):
    def __init__(self, data_matrix, df, site_colname, covariate_cols=None):
        super().__init__(data_matrix, df, site_colname, covariate_cols)

    def _harmonize(self):
        categorical_cols = []
        continuous_cols = []

        for cc in self.covariate_cols:
            if is_numeric_dtype(self.df[cc]):
                continuous_cols.append(cc)
            elif is_string_dtype(self.df[cc]) or is_categorical_dtype(self.df[cc]):
                categorical_cols.append(cc)
            else:
                msg = f"column {cc} is not continuous, categorical" \
                "or string. Ignoring..."
                warnings.warn(msg)

        return neuroCombat(
            dat=self.data_matrix,
            covars=self.df,
            batch_col=self.site_colname,
            categorical_cols=categorical_cols,
            continuous_cols=continuous_cols,
        )