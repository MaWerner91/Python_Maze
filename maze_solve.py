""" short script to exemplify the use of PerfectMaze """

# suppress UPPER_CASE naming style errors
# pylint: disable=C0103

import sys
from maze import PerfectMaze

if __name__ == "__main__":

    if len(sys.argv) > 1:
        N = int(sys.argv[1])
    else:
        N = 20

    perfect_maze = PerfectMaze(N)
    perfect_maze.show_path(0, N**2-1)
