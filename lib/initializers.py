from mimetypes import init
import torch

def get(identifier):
    """Retrieve an initializer by the identifier.

    Args:
        identifier: String that contains the initializer name or an initializer
            function.

    Returns:
        Initializer instance base on the input identifier.
    """
    initializer_dict = {
        "Glorot normal": torch.nn.init.xavier_normal_,
        "Glorot uniform": torch.nn.init.xavier_uniform_,
        "He normal": torch.nn.init.kaiming_normal_,
        "He uniform": torch.nn.init.kaiming_uniform_,
        "zeros": torch.nn.init.zeros_,
    }
    if isinstance(identifier, str):
        return     initializer_dict[identifier]
    if callable(identifier):
        return identifier
    raise ValueError("Could not interpret initializer identifier: " + str(identifier))