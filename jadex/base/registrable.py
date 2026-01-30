from typing import Dict, List, Optional


class Registrable:
    # NOTE: This must be added to the the parent class as a class variable:
    # registered: ClassVar[Dict[str, Type[Self]]] = dict()

    @classmethod
    def get_name(cls) -> str:
        return cls.__name__

    @classmethod
    def register(cls):
        cls_name = cls.get_name()

        if cls_name in cls.registered:
            raise ValueError(f"'{cls_name}' is already registered.")

        cls.registered[cls_name] = cls

    @classmethod
    def already_registered(cls):
        return len(cls.list_registered()) > 0

    @classmethod
    def list_registered(cls) -> List[str]:
        return list(cls.registered.keys())


def register_all():
    from jadex.algorithms import register_algorithms
    from jadex.data import register_data_classes
    from jadex.distributions import register_distributions
    from jadex.downstream import register_downstream
    from jadex.networks import register_networks

    register_algorithms()
    register_data_classes()
    register_distributions()
    register_networks()
    register_downstream()
