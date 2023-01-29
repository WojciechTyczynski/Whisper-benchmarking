import os

import hydra
from hydra import compose

hydra.initialize_config_dir(config_dir=f'{os.getcwd()}/conf')

api_config = compose(config_name='conf')
