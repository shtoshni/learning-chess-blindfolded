from os import path


def get_eval_set_name(filename):
    name = path.basename(filename)
    name = path.splitext(name)[0]
    return ' '.join([part.capitalize() for part in name.split('_')])
