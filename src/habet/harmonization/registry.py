import pkg_resources


_registry = dict()

def load_endpoints():
    for entry_point in pkg_resources.iter_entry_points("habet.harmonization_methods"):
        entry_point.load()

def register_harmonization_method(cls):
    _registry[cls.__name__] = cls

def get_registry_dict():
    return _registry.copy()