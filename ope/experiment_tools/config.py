

class Config(object):
    def __init__(self, cfg):
        self._config = cfg # set it to conf

    def __getattr__(self, property_name):
        if property_name not in self._config.keys(): # we don't want KeyError
            return None  # just return None if not found
        return self._config[property_name]

    def add(self, cfg):
        self._config.update(cfg)
