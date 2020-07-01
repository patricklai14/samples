import pdb

import numpy as np
import pandas as pd

from bball import constants

def transform_features(X):
    add_cols = [constants.PPG_COL, constants.PPG_PREV1_COL, constants.PPG_PREV2_COL]
    multiply_cols = [constants.HOME_NOT_AWAY_COL, constants.OPP_ALLOWED_RATING_COL]

    new_features = []
    for add_col in add_cols:
        for multiply_col in multiply_cols:
            new_features.append(X[add_col] * X[multiply_col])
    return np.array(new_features).transpose()

if __name__ == "__main__":
    X = pd.DataFrame({constants.PLAYER_ID_COL: ["bob", "bob", "bob"], constants.PPG_COL: [20., 22.5, 18.3], 
                         constants.PTS_COL: [20., 25., 10.], constants.OPP_ALLOWED_RATING_COL: [1.01, 0.95, 0.98],
                         constants.HOME_NOT_AWAY_COL: [1, 1, 0], constants.PPG_PREV1_COL: [20., 20., 20.],
                         constants.PPG_PREV2_COL: [18., 18., 18.]})
    X_new = transform_features(X)
    pdb.set_trace()