from pathlib import Path
from tqdm import tqdm

# TODO: implement this method
def read_pairs(path: str):
    """
    Read tab-delimited pairs from file.
    Parameters
    ----------
    path: str 
        path to the input file
    Returns
    -------
        a list of pair tuple
    """
    # BEGIN SOLUTION
    with open(path, "r") as file:
        return [tuple(line.strip().split("\t")) for line in file]
    # END SOLUTION


# TODO: implement this method
def read_triplets(path: str):
    """
    Read tab-delimited triplets from file.
    Parameters
    ----------
    path: str 
        path to the input file
    Returns
    -------
        a list of triplet tuple
    """
    # BEGIN SOLUTION
    with open(path, "r") as file:
        return [tuple(line.strip().split("\t")) for line in file]
    # END SOLUTION