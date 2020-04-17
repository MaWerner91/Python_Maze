""" short script to compute forward committor in a PerfectMaze """

# suppress UPPER_CASE naming style errors
# pylint: disable=C0103

import argparse
import numpy as np
import scipy as sp
from scipy.sparse.linalg import spsolve
from maze import PerfectMaze

parser = argparse.ArgumentParser(description="Generate N-by-N perfect maze and compute \
                                the forward committor from bottom left to top right corner")
parser.add_argument("-N",
                    metavar="N", default=20,
                    type=int, help="N-by-N perfect maze (default: 20)")

if __name__ == "__main__":

    args = parser.parse_args()
    N = args.N

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
