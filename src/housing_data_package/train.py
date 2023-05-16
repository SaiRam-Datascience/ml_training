import argparse
import logging
import pickle
import sys
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

PARAM_GRID = {
    "n_estimators": randint(low=1, high=50),
    "max_features": randint(low=1, high=8),
    "max_depth": randint(low=1, high=20),
    "criterion": ["mse", "poisson"],
}


class HyperParameterTuning:
    """A utility class to do Hyperparameter Tuning using RandomisedSearchCV"""

    def __init__(self, training_data: np.ndarray, training_labels: np.ndarray):
        """
        __init__ Initialize the datasets to do hyperparameter tuning
        Parameters
        ----------
        training_data :  np.ndarray
            the training data consisting of independent features
        training_labels :  np.ndarray
            the training data consisting of target values
        """
        self.training_data = training_data
        self.training_labels = training_labels

    def find_best_hyperparameters(self, max_evals: int = 100) -> RandomForestRegressor:
        """
        find_best_hyperparameters A helper utility to find best hyperparameters
        Parameters
        ----------
        max_evals :  int, optional
            maximum number of hyperparameters combinations to try, by default 100
        Returns
        -------
        RandomForestRegressor
            returns a RandomForestRegressor model which is fine tuned
        """
        forest_reg = RandomForestRegressor(random_state=42, n_jobs=-1)
        rnd_search = RandomizedSearchCV(
            forest_reg,
            param_distributions=PARAM_GRID,
            n_iter=max_evals,
            cv=5,
            scoring="neg_mean_squared_error",
            random_state=42,
        )
        rnd_search.fit(self.training_data, self.training_labels)
        final_model = rnd_search.best_estimator_
        return final_model


def data_split_target_labels(df: pd.DataFrame, target_column: str) -> Tuple[np.ndarray]:
    """
    data_split_target_labels A helper function to seperate Independent and target column
    Parameters
    ----------
    df :  pd.DataFrame
        the dataframe to separate Independent and dependent features
    target_column :  str
        the name of target values in the df dataframe
    Returns
    -------
    Tuple[np.ndarray]
        returns Independent and dependent features as np.ndarray's
    """
    X = df.drop(columns=target_column, axis=1).values
    y = df[target_column].values
    return X, y


def main() -> None:
    """
    The Core Logic to train and finetune random forest model resides here
    """
    parser = argparse.ArgumentParser(
        description="Train Random Forest Regressor and store the Best Model"
    )
    parser.add_argument(
        "-trdp",
        "--training_data_path",
        help="Give the Input data file path to train the model",
        default="../../data/processed/train/train.csv",
        # default="../../data/train.csv",
    )
    parser.add_argument(
        "-trmp",
        "--trained_model_path",
        help="Give the Output file path to store the Best Model in a pickle format",
        default="../../artifacts/best_random_forest_model.pkl",
        # default = "../../artifacts/regressor.pkl"
    )

    parser.add_argument(
        "-l",
        "--log_level",
        choices=["debug", "info", "error"],
        help="Set up the log level",
        default="debug",
    )
    parser.add_argument(
        "-lp",
        "--log_file_path",
        help="""Specify the folder path into which log files must be \
                 written by defualt log files will not be generated""",
        default=None,
    )
    parser.add_argument(
        "-ncl",
        "--no_console_log",
        action="store_true",
        help="Toggle weather to write logs to console",
    )
    args = parser.parse_args()
    training_data_path = args.training_data_path
    trained_model_path = args.trained_model_path
    log_level = args.log_level
    log_file_path = args.log_file_path
    console_log = args.no_console_log

    logFormatter = logging.Formatter("'%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    rootLogger = logging.getLogger()

    if log_level == "debug":
        rootLogger.setLevel(logging.DEBUG)
    elif log_level == "error":
        rootLogger.setLevel(logging.ERROR)
    else:
        rootLogger.setLevel(logging.INFO)

    if log_file_path:
        fileHandler = logging.FileHandler("{0}/train_model.log".format(log_file_path))
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)

    if not console_log:
        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setFormatter(logFormatter)
        rootLogger.addHandler(consoleHandler)
    rootLogger.info("Preparing the Training Data..")

    training_data = pd.read_csv(training_data_path)
    X_tr, y = data_split_target_labels(training_data, "median_house_value")

    rootLogger.info("Training RandomForest Regressor with RandomizedSearchCV")

    hyperparameter_tuner = HyperParameterTuning(X_tr, y)
    final_model = hyperparameter_tuner.find_best_hyperparameters(10)

    rootLogger.info("Saving the best RandomForest Regressor Model...")

    with open(trained_model_path, "wb") as file_handler:
        pickle.dump(final_model, file_handler)

    rootLogger.info(f"Saved the best RandomForest Regressor Model at {trained_model_path}")


if __name__ == "__main__":
    main()
