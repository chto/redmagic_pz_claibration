import oyaml as yaml
from collections import OrderedDict
import os
def chto_yamlload(filename, parent_dir=None):
    with open(filename, 'r') as stream:
        try:
            param =  yaml.load(stream)
        except yaml.YAMLError as exc:
            print("yaml loaderr")
            print(exc)

    for inc in param.get("include", []):
        if parent_dir is not None:
            inc = os.path.join(parent_dir, inc)
        test = chto_yamlload(inc)
        test.update(param)
        param = test
    return param
