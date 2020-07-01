This is a tool for predicting NBA player scoring outputs on a game-to-game basis. The project currently contains the following components:
- (Code for generating) custom-built dataset of predictive signals/features, e.g. opponent-adjusted defensive metrics, scoring by play type, etc.
- Library of regression/ML models for prediction
- Testing/evaluation framework for back-testing models and simulating model performance

Currently under development:
- Additional predictive signals/features:
    - Position-adjusted defensive metrics
    - Individual matchup data
- Additional regression models:
    - Neural networks
    - Boosted algorithms
- Feature selection/regularization step - using lasso/elastic net for feature selection, as some of the existing features are highly correlated


This project uses the following packages:
- numpy
- pandas
- scikit-learn
- basketball_reference_web_scraper