from v2a.configs.config import Config
from v2a.configs.base_config import config_factory, get_all_registered_configs

# note: these imports are needed to register these classes in the global config registry
from v2a.configs.v2a_config import V2AConfig