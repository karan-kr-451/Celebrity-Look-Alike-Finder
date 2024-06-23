import os
import argparse
from src import logger
from src.utils.utils import read_yaml, create_directory, read_names_from_txt, data_download

def Image_Downloader(config_path,params_path):
    
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    artifacts = config['artifacts']

    artifacts_dir = artifacts['artifacts_dir']
    upload_image_dir = artifacts['upload_image_dir']
    calebs_name = artifacts['celeb_name']

    raw_local_dir_path = os.path.join(artifacts_dir, upload_image_dir)
    create_directory(dirs=[raw_local_dir_path])

    # dataset = os.path.join(raw_local_dir_path, upload_image_dir)
    names = read_names_from_txt(calebs_name)
    limits = params['base']['limits']

    for name in names:
        data_download(name, limits, raw_local_dir_path)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()
    
    try:
        logger.info(">>>>> stage_02 started")
        generate_data_pickle_file(config_path = parsed_args.config, params_path= parsed_args.params)
        logger.info("stage_02 completed!>>>>>")
    except Exception as e:
        logger.exception(e)
        raise e