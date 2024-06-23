import argparse
from src import logger
import sys
from src.Exception import CelebsException
from src.pipeline.dataloader import Image_Downloader
from src.pipeline.image_pickle import generate_data_pickle_file
from src.pipeline.feature_extractor import feature_extractor
from src.pipeline.DataCleaner import DataClean




if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logger.info(">>>>> stage_01 started")
        Image_Downloader(config_path = parsed_args.config, params_path= parsed_args.params)
        logger.info("stage_01 completed!>>>>>")

        logger.info(">>>>> stage_02 started")
        DataClean(config_path=parsed_args.config, params_path=parsed_args.params)
        logger.info("stage_2 completed!>>>>>")

        logger.info(">>>>> stage_03 started")
        generate_data_pickle_file(config_path=parsed_args.config, params_path=parsed_args.params)
        logger.info("stage_03 completed!>>>>>")

        logger.info(">>>>> stage_04 started")
        feature_extractor(config_path=parsed_args.config, params_path=parsed_args.params)
        logger.info("stage_4 completed!>>>>>")
    except Exception as e:
        raise CelebsException(e, sys)