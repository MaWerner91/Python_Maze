import sys
from maze import PerfectMaze

if len(sys.argv) > 1:
    N = int(sys.argv[1])
else:
    N = 20

m = PerfectMaze(N)
m.show_path(0, N**2-1)
