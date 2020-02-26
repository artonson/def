def load_func_from_config(func_dict, config):
    return func_dict[config['type']].from_config(config)
