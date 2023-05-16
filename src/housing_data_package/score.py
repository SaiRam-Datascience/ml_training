import argparse
import logging
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from housing_data_package.train import data_split_target_labels


def main():
    """The core logic to score the model resides here"""
    parser = argparse.ArgumentParser(
        description="Score on Test Data using Best Trained Random Forest Regressor"
    )
    parser.add_argument(
        "-tedp",
        "--test_data_path",
        help="Give the Input data file path to test the model",
        default="../../data/processed/test/test.csv",
    )
    parser.add_argument(
        "-trmp",
        "--trained_model_path",
        help="Give the Input file path of the stored Best Model in a pickle format",
        default="../../artifacts/best_random_forest_model.pkl",
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
    testing_data_path = args.test_data_path
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
        fileHandler = logging.FileHandler("{0}/score_model.log".format(log_file_path))
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)

    if not console_log:
        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setFormatter(logFormatter)
        rootLogger.addHandler(consoleHandler)

    rootLogger.info("Fetching the Test Data to score the Best Trained Random Forest Model..")

    test_data = pd.read_csv(testing_data_path)
    X_te, y = data_split_target_labels(test_data, "median_house_value")

    rootLogger.info("Loading the Best Trained Random Forest Regressor Model..")

    with open(trained_model_path, "rb") as f:
        best_model = pickle.load(f)

    rootLogger.info("Scoring the data on Best Trained Random Forest Model..")
    preds = best_model.predict(X_te)
    test_mse = mean_squared_error(y, preds)
    test_rmse = np.sqrt(test_mse)

    rootLogger.info(f"RMSE on Test Data with Best Trained Random Forest Model is {test_rmse}")


if __name__ == "__main__":
    main()
