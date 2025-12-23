# Database Contents License (DbCL)
# Copyright (C) 2025 JustSplash8501
# Licensed under the Database Contents License (DbCL).
# The contents are provided "as is" without warranty of any kind.

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

def compute_vif(df):
    numeric_cols = df.select_dtypes(include="number")

    numeric_cols = add_constant(numeric_cols)
    vif = pd.DataFrame(
        {
            "feature": numeric_cols.columns,
            "VIF": [
                variance_inflation_factor(numeric_cols.values, i)
                for i in range(numeric_cols.shape[1])
            ],
        }
    )
    return vif

def adj_r2_score(r2, n, p):
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return adj_r2

