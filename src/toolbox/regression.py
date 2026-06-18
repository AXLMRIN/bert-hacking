import pandas as pd 
import numpy as np 
import statsmodels.api as sm

def perform_regression(
    y_column: pd.Series, 
    x_column: pd.Series, 
    optimizer: str = "bfgs"
) -> dict[str:str|float|list[float]]:
    """
    Perform logit regression and returns key values
    """
    Y = y_column.to_numpy().astype(int) # [1, 0, 0, 1, ...]
    X = x_column.to_numpy().astype(int) # [1, 0, 1, 0, ...]
    X = sm.add_constant(X)
    try: 
        model = sm.Logit(Y,X, )
        res = model.fit(maxiter=100, method=optimizer)

        return {
            "success":True,
            "optimizer": optimizer,
            "Pseudo R-squared": res.prsquared,
            "Covariate Names": model.exog_names,
            "Coef": res.params.tolist(),
            "Std err": res.bse.tolist(),
            "z": res.tvalues.tolist(),          # z-statistics
            "pvalues": res.pvalues.tolist(),
            "Conf Int": res.conf_int().tolist(),
            "Log-Likelihood": res.llf,
            "LL-Null": res.llnull,
            "LLR p-value": res.llr_pvalue,
            "AIC": res.aic,
            "BIC": res.bic,
            "N obs": res.nobs,
        }
    except Exception as e: 
        return {
            "success":False,
            "optimizer": optimizer,
            "Pseudo R-squared": None,
            "Covariate Names": None, 
            "Coef": None,
            "Std err": None,
            "z": None,          # z-statistics
            "pvalues": None,
            "Conf Int": None,
            "Log-Likelihood": None,
            "LL-Null": None,
            "LLR p-value": None,
            "AIC": None,
            "BIC": None,
            "N obs": None,
        } 
