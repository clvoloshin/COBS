from ope.models.conv import *

def get_model_from_name(name):
    if name == 'tabular':
        return 'tabular'
    elif name == 'defaultCNN':
        return defaultCNN
    elif name == 'defaultModelBasedCNN':
        return defaultModelBasedCNN
    else:
        raise NotImplemented

def setup_params(param):
    # replace string of model with model itself in the configuration.
    for method, parameters in param['models'].items():
        param['models'][method]['model'] = get_model_from_name(parameters['model'])
    
    param['experiment']['to_regress_pi_b']['model'] = get_model_from_name(param['experiment']['to_regress_pi_b']['model'] )

    return param
