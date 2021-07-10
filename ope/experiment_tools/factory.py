from ope.models.conv import defaultCNN

def get_model_from_name(name):
    if name == 'defaultCNN':
        return defaultCNN
