import os.path as path


def to_absolute_path(p: str, file_path: str) -> str:
    return path.normpath(path.join(path.dirname(file_path), p))
