import argparse
import logging
import os
import sys
import tarfile

import numpy as np
import pandas as pd
from six.moves import urllib  # type: ignore
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit


class Download_Dataset:
    """
    This class helps is downloading the data and storing it in respective path

    """

    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

    def __init__(self, raw_data_path: str) -> None:
        """
        __init__ Initialize the raw data variable

        Parameters
        ----------
        raw_data_path : str
            An input taken from command line stores the path information to stores the raw data
        """
        self.raw_data_path = raw_data_path

    def fetch_housing_data(self, housing_url: str = HOUSING_URL) -> None:
        """
        fetch_housing_data This helper function downloads the that from URL and store the data

        Parameters
        ----------
        housing_url : str, optional
            This variable contains the URL info from where data needs to be downloaded by default
            HOUSING_URL
        """
        os.makedirs(self.raw_data_path, exist_ok=True)
        tgz_path = os.path.join(self.raw_data_path, "housing.tgz")
        urllib.request.urlretrieve(housing_url, tgz_path)
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path=self.raw_data_path)
        housing_tgz.close()

    def load_housing_data(self) -> pd.DataFrame:
        """
        load_housing_data load_housing_data This helper function loads the raw data

        Returns
        -------
        pd.DataFrame
            Returns the pandas dataframe containing information of raw data.
        """
        csv_path = os.path.join(self.raw_data_path, "housing.csv")
        return pd.read_csv(csv_path)


class Split_Train_Test:
    """
    This class helps in separating train and test class from raw data
    """

    def __init__(self, df_data: pd.DataFrame) -> None:
        """
        __init__ Initialize the raw dataframe to be splitted into train and test

        Parameters
        ----------
        df_data : pd.DataFrame
            Taking input as raw dataframe
        """
        self.df_data = df_data

    def splitter(self) -> tuple:
        """
        splitter This helper function splits the data into train and test

        Returns
        -------
        tuple
            returns a tuple containing train and test dataset
        """
        self.df_data["income_cat"] = pd.cut(
            self.df_data["median_income"],
            bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
            labels=[1, 2, 3, 4, 5],
        )

        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(self.df_data, self.df_data["income_cat"]):
            strat_train_set = self.df_data.loc[train_index]
            strat_test_set = self.df_data.loc[test_index]

        for set_ in (strat_train_set, strat_test_set):
            set_.drop("income_cat", axis=1, inplace=True)

        return (strat_train_set, strat_test_set)


class Feature_Transformation:
    """
    This function helps in doing feature engineering
    """

    def __init__(self, df_data: pd.DataFrame) -> None:
        self.df_data = df_data

    def get_new_features(
        self,
        numerator_col: str,
        denominator_col: str,
        result_col: str,
    ) -> pd.DataFrame:
        """
        get_new_features

        Parameters
        ----------
        result_col : str
            new column name
        numerator_col : str
            numerator column of the dataframe
        denominator_col : str
            denominator column of the dataframe

        Returns
        -------
        pd.DataFrame
            returns the transformed new features
        """
        self.df_data[result_col] = self.df_data[numerator_col] / self.df_data[denominator_col]

        return self.df_data


def main():

    """
    This is the main functions which connects all the helper functions
    """

    parser = argparse.ArgumentParser(description="Download and Store the Raw Data")
    parser.add_argument(
        "-r",
        "--raw_data_path",
        help="Give the Output folder path to store the raw data",
        default="../../data/raw",
    )

    parser.add_argument(
        "-p",
        "--processed_data_path",
        help="Give the Output folder path to store the processed data into train and test data",
        default="../../data/processed",
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
        default="../../logs",
    )
    parser.add_argument(
        "-ncl",
        "--no_console_log",
        action="store_true",
        help="Toggle weather to write logs to console",
    )

    args = parser.parse_args()
    raw_data_path = args.raw_data_path
    processed_path = args.processed_data_path
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
        fileHandler = logging.FileHandler("{0}/ingest_data.log".format(log_file_path))
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)

    if not console_log:
        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setFormatter(logFormatter)
        rootLogger.addHandler(consoleHandler)

    rootLogger.info("Fetching the Housing Data from External URL..")

    get_data = Download_Dataset(raw_data_path)
    get_data.fetch_housing_data()
    housing = get_data.load_housing_data()

    rootLogger.info("Splitting the data into Train and Test")
    splitter_obj = Split_Train_Test(housing)
    strat_train_set, strat_test_set = splitter_obj.splitter()

    housing_tr = strat_train_set.drop("median_house_value", axis=1)
    housing_labels_tr = strat_train_set["median_house_value"].copy()

    housing_te = strat_test_set.drop("median_house_value", axis=1)
    housing_labels_te = strat_test_set["median_house_value"].copy()

    rootLogger.info("Performing Imputation")

    imputer = SimpleImputer(strategy="median")

    housing_tr_num = housing_tr.drop("ocean_proximity", axis=1)
    housing_te_num = housing_te.drop("ocean_proximity", axis=1)

    imputer.fit(housing_tr_num)
    X_tr = imputer.transform(housing_tr_num)
    X_te = imputer.transform(housing_te_num)

    housing_tr_final = pd.DataFrame(X_tr, columns=housing_tr_num.columns, index=housing_tr.index)
    housing_te_final = pd.DataFrame(X_te, columns=housing_te_num.columns, index=housing_te.index)
    rootLogger.info("Performing the feature Transformation")
    ft_tr_obj = Feature_Transformation(housing_tr_final)

    housing_tr_final = ft_tr_obj.get_new_features(
        "total_rooms", "households", "rooms_per_household"
    )
    housing_tr_final = ft_tr_obj.get_new_features(
        "total_bedrooms", "total_rooms", "bedrooms_per_room"
    )
    housing_tr_final = ft_tr_obj.get_new_features(
        "population", "households", "population_per_household"
    )

    ft_te_obj = Feature_Transformation(housing_te_final)

    housing_te_final = ft_te_obj.get_new_features(
        "total_rooms", "households", "rooms_per_household"
    )
    housing_te_final = ft_te_obj.get_new_features(
        "total_bedrooms", "total_rooms", "bedrooms_per_room"
    )
    housing_te_final = ft_te_obj.get_new_features(
        "population", "households", "population_per_household"
    )

    housing_tr_cat = housing_tr[["ocean_proximity"]]
    housing_tr_final = housing_tr_final.join(pd.get_dummies(housing_tr_cat, drop_first=True))

    housing_te_cat = housing_te[["ocean_proximity"]]
    housing_te_final = housing_te_final.join(pd.get_dummies(housing_te_cat, drop_first=True))

    housing_tr_final["median_house_value"] = housing_labels_tr
    housing_te_final["median_house_value"] = housing_labels_te

    rootLogger.info("Saving the Processed Train and Test Data...")

    os.makedirs(os.path.join(processed_path, "train"), exist_ok=True)

    housing_tr_final.to_csv(
        os.path.join(processed_path, "train", "train.csv"),
        index=False,
        index_label=False,
    )

    os.makedirs(os.path.join(processed_path, "test"), exist_ok=True)
    housing_te_final.to_csv(
        os.path.join(processed_path, "test", "test.csv"),
        index=False,
        index_label=False,
    )


if __name__ == "__main__":
    """
    This function calls the main function
    """
    main()
