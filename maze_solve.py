""" short script to exemplify the use of PerfectMaze """

# suppress UPPER_CASE naming style errors
# pylint: disable=C0103

import argparse
from maze import PerfectMaze

parser = argparse.ArgumentParser(description="Generate N-by-N perfect maze and show \
                                the path from bottom left top right corner")
parser.add_argument("-N",
                    metavar="N", default=20,
                    type=int, help="N-by-N perfect maze (default: 20)")

if __name__ == "__main__":

    args = parser.parse_args()

    perfect_maze = PerfectMaze(args.N)
    perfect_maze.show_path(0, args.N**2-1)
