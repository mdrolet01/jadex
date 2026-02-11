import hydra
from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin


class JadexConfigPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        # Make configs in jadex/global_configs accessible from any directory
        search_path.append(provider="jadex", path="pkg://jadex/global_configs/")


def register_jadex_plugin() -> None:
    """Hydra users should call this function before invoking @hydra.main"""
    Plugins.instance().register(JadexConfigPlugin)


def jadex_hydra_main(config_name="", config_path="."):
    """
    Custom decorator that wraps hydra.main and automatically registers resolvers.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            register_jadex_plugin()

            ##### Register any other hydra plugins/packages here #####

            return hydra.main(str(config_path), config_name, version_base=None)(func)(*args, **kwargs)

        return wrapper

    return decorator
