import os

def get_folders(path):
    """
    Returns a list of all folders in the specified path.
    
    Parameters:
        path (str): The path to search for folders.
        
    Returns:
        list: A list of folder names in the given path.
    """
    if not os.path.isdir(path):
        raise ValueError("The provided path is not a directory.")
        
    folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    return folders

