from src.textSummarizer.constants import *
from src.textSummarizer.utils.common import read_yaml, create_directories

from src.textSummarizer.entity.config_entity import DataIngestionConfig

class ConfigurationManager:
    def __init__(self,
                 config_path=CONFIG_FILE_PATH,
                 params_filepath=PARAMS_FILE_PATH):
        self.config=read_yaml(config_path)
        self.paramss=read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            dataset_name=config.dataset_name,
            save_path=Path(config.save_dir)
        )