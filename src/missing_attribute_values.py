#!/usr/bin/python2

import sys
import os

def get_features_with_missing_data(physics_data):
    """ Determines which columsn contain missing data

    Arguments:
    physics_data -- An Nx80 list of lists containing physics data.

    Returns:
    columns_unique -- A set of column values (8 of them) that are
    currently missing data (represented as 999.0 or 9999.0

    We are interested in columns 2:79, which contain attribute values.
    I'm pretty sure there is a function in numpy that does this, too,
    and probably faster, but this is fine for now.
    """

    # First find columns that contain 999 or 9999
    find_columns = lambda ns, row : ([i for i,e in enumerate(row) if e in ns])

    # We know that missing data is noted as 999 or 9999 but not which
    # columns contain it.
    missing = [999.0, 9999.0]

    # Missing value columns
    columns = [find_columns(missing,row) for row in physics_data]
    columns_unique = set(item for sublist in columns for item in sublist)
    columns_unique.discard(0)

    return [c - 2 for c in columns_unique]


def read_physics_data(filename):
    """ Reads in particle physics data.

    Arguments:
    filename -- data file.

    Column 0: 1-indexed datapoint #
    Column 1: Binary classification, 1 or 0
    Columns 2-79: 78 features, unlabelled
    """

    # Open filename for reading
    f = open(filename, 'r')

    # Split on whitespace, creating multi-dimensional array
    data = [map(float, line.split()) for line in f]

    # We're done now
    f.close()
    return data

if __name__ == "__main__":
    # Read file provided into list of lists
    raw_data = read_physics_data(sys.argv[1])

    # Get columns with missing features - expect 8
    missing_data_columns = get_features_with_missing_data(raw_data)

    # Just print missing columns for now"
    print "nth features containing missing data:\n",\
        missing_data_columns,\
        "\nAdd 2 to values to get Nx80 dataset column position"
