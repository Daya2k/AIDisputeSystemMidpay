import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
# inherits some special methods like __init__,etc
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation, DataTransformationConfig


@dataclass
class DataIngestionConfig:
    '''
    To save data in the specified path
    '''
    train_data_path: str = os.path.join('artifacts', 'train.pkl')
    test_data_path: str = os.path.join('artifacts', 'test.pkl')
    raw_data_path: str = os.path.join('artifacts', 'data.parquet')


class DataIngestion:
    '''Acquiring data from sources'''

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        '''Data Ingestion'''
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_parquet('notebooks/data/merged_data.parquet')
            logging.info('read the parquet dataset as dataframe')

            os.makedirs(os.path.dirname(
                self.ingestion_config.train_data_path), exist_ok=True)

            df.to_parquet(self.ingestion_config.raw_data_path)

            logging.info("Train Test Split Started")
            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=42, stratify=df.classes.values)

            train_set.to_pickle(self.ingestion_config.train_data_path)
            test_set.to_pickle(self.ingestion_config.test_data_path)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_df, test_df = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_df, test_df)
