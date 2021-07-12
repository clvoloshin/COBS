from ope.models.conv import *

def get_model_from_name(name):
    if name == 'defaultCNN':
        return defaultCNN
    elif name == 'defaultModelBasedCNN':
        return defaultModelBasedCNN
    else:
        raise NotImplemented
