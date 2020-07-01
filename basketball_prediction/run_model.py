import argparse
import pdb

import pandas as pd

from basketball_reference_web_scraper import client, data

from bball import stats, predictor, preprocessor

def main():
    # #write data
    # stats.write_data()

    #get data
    train_data = stats.read_data(2003, 2019)

    #fit model
    model = predictor.predictor("linear")
    X, y = train_data.get_data()
    X_transformed = preprocessor.transform_features(X)
    model.fit(X_transformed, y)

    #make prediction
    #['ppg', 'ppg_prev1', 'ppg_prev2', 'opp_allowed_rating', 'home_not_away']
    players = ['butleji01']
    inputs = stats.get_player_current_stats(players)
    pdb.set_trace()

    inputs = pd.DataFrame([[15.5, 12.3, 14.0, 0.961121157323689, 0]], columns=train_data.features)
    inputs_transformed = preprocessor.transform_features(inputs)
    prediction = model.predict(inputs_transformed)
    pdb.set_trace()

if __name__ == "__main__":
    main()