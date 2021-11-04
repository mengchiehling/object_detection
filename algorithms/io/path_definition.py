import os

def get_project_dir() -> str:

    """
    Get the full path to the repository
    """

    dir_as_list = os.path.dirname(__file__).split("/")
    index = dir_as_list.index("algorithms")
    project_directory = f"/{os.path.join(*dir_as_list[:index + 1])}"

    return project_directory


def get_file(relative_path: str) -> str:

    """
    Given the relative path to the repository, return the full path
    """

    project_directory = get_project_dir()
    return os.path.join(project_directory, relative_path)
