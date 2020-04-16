""" short script to compute forward committor in a PerfectMaze """

# suppress UPPER_CASE naming style errors
# pylint: disable=C0103

import sys
import numpy as np
import scipy as sp
from scipy.sparse.linalg import spsolve
from maze import PerfectMaze

if __name__ == "__main__":

    if len(sys.argv) > 1:
        N = int(sys.argv[1])
    else:
        N = 20

    perfect_maze = PerfectMaze(N)

    # compute committor as in
    # https://github.com/markovmodel/msmtools/blob/devel/msmtools/analysis/sparse/committor.py

    # generator matrix
    L = perfect_maze.T - sp.sparse.eye(perfect_maze.T.shape[0], perfect_maze.T.shape[1])

    # incorporate boundary conditions
    L[0] = 0.
    L[0, 0] = 1.
    L[N**2-1] = 0.
    L[N**2-1, N**2-1] = 1.

    # set up right hand side
    rhs = np.zeros(N**2)
    rhs[-1] = 1.

    committor = spsolve(L, rhs)

    perfect_maze.visualize(committor=committor)
