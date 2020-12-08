import os

def get_directory(file):
    return os.path.dirname(file)

def abspath_from_file(file, relative_path):
    return os.path.join(os.path.dirname(os.path.abspath(file)), relative_path)