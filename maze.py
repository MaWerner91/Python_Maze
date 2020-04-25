"""This module contains a class to generate and solve a perfect maze"""

# suppress snake_case naming style errors
# pylint: disable=C0103

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

class PerfectMaze():

    """ Class for generating and solving a N-by-N perfect maze
    This class generates a perfect maze by using union find. The maze
    can be visualized, as well as the commitor, and can be solved
    using recursive tree search. The states are enumarated from 0 to
    N^2-1.

    Parameters:

        N: (int) for N-by-N maze

    Example:

        from maze import PerfectMaze

        maze = PerfectMaze(20)
        maze.show_path(0, 399)

    Attributes:

        N: for N-by-N maze
        walls_horizontal: list of states, which have a horizontal wall
        walls_vertical: list of states, which have a vertical wall
        T: sparse transition matrix

    Methods:

        show_path(start, end):
            Plots the maze and indicates the path between state start and state end

            Parameters:
                start: (int) start state
                end: (int) end state
            Returns:
                path: (ndarray) sequence of states that connect start/end

        random_walk(steps, start, end=None):
            Generates and plots a random walk though the maze

            Parameters:
                steps: (int) number of steps of random walk
                start: (int) start state
                end: (int) end state

            Returns:
                rand_walk: (ndarray) sequences of states from random walk

        visualize(wait=False, labels=False, committor=None):
            Prints the maze

            Parameters:
                wait: (bool) if True, plt.show() is not called by method
                labels: (bool) if True, walls will be labeled with respective number
                committor: (ndarray) color plot of commitor in maze

            Returns:
                nothing

    """
    def __init__(self, N):

        # some initialisations
        self.N = N
        self.__sets = [np.array([i]) for i in range(N*N)]
        self.walls_vertical = []
        self.walls_horizontal = []

        # create lists of the walls
        for i in range(N**2):
            if i <= N**2-N-1:
                self.walls_vertical.append(i)
            if np.mod(i, N) != N-1:
                self.walls_horizontal.append(i)

        # shuffle list of walls
        np.random.shuffle(self.walls_vertical)
        np.random.shuffle(self.walls_horizontal)

        # wall counter variables
        whc = 0
        wvc = 0

        # remove walls and join __sets
        while len(self.__sets) > 1:
            # states that will be connected
            c1 = 0
            c2 = 0
            # flag to indicate which direction was removed
            dirflag = 0

            # pick a wall to remove and determine neighbours
            u = np.random.rand()
            if len(self.walls_horizontal) == 0:
                c1 = self.walls_vertical[wvc]
                c2 = c1 + self.N
                dirflag = 1
            elif len(self.walls_vertical) == 0:
                c1 = self.walls_horizontal[whc]
                c2 = c1 + 1
            elif u < 0.5:
                c1 = self.walls_vertical[wvc]
                c2 = c1 + self.N
                dirflag = 1
            elif u >= 0.5:
                c1 = self.walls_horizontal[whc]
                c2 = c1 + 1
            else:
                print("All walls removed!")
                break

            s1 = 0
            s2 = 0

            # find sets with c1, c2
            for i in range(len(self.__sets)):
                if c1 in self.__sets[i]:
                    s1 = i
                if c2 in self.__sets[i]:
                    s2 = i

            # if neighbours aren't in same set => merge sets, remove wall
            if s1 != s2:
                self.__sets[s1] = np.concatenate((self.__sets[s1], self.__sets[s2]))
                if dirflag == 0:
                    self.walls_horizontal.remove(c1)
                else:
                    self.walls_vertical.remove(c1)
                removearray(self.__sets, self.__sets[s2])
            else:
                if dirflag == 0:
                    whc += 1
                else:
                    wvc += 1

        # set up sparse transition matrix
        col = []
        row = []
        data = []
        for state in range(self.N**2):
            options = 1
            col.append(state)
            row.append(state)

            # consider all walls and count move options
            if state not in self.walls_horizontal and np.mod(state+1, N) != 0:
                options += 1
                col.append(state+1)
                row.append(state)
            if state not in self.walls_vertical and state+self.N < self.N**2:
                options += 1
                col.append(state+self.N)
                row.append(state)
            if state-1 not in self.walls_horizontal and np.mod(state, N) != 0:
                options += 1
                col.append(state-1)
                row.append(state)
            if state-self.N not in self.walls_vertical and state-self.N >= 0:
                options += 1
                col.append(state-self.N)
                row.append(state)

            # add possible moves to transition matrix with equal probability
            for i in range(options):
                data.append(1/options)

        self.T = sp.coo_matrix((data, (row, col)), shape=(N**2, N**2))

    def show_path(self, start, end):
        """ show path between start and end """

        # remove diagonal elements from transition matrix
        row, col = self.T.nonzero()
        while (row == col).any():
            diags = np.where(row == col)[0][0]
            row = np.delete(row, diags)
            col = np.delete(col, diags)

        # recursively find path
        path = self.__path_search(row, col, start, end, start)
        self.visualize(wait=True)

        # plot path
        for i in range(len(path)-1):
            x0 = np.mod(path[i], self.N)
            y0 = (path[i]-x0)/self.N
            if np.abs(path[i] - path[i+1]) <= 1:
                x1 = x0 - path[i] + path[i+1]
                y1 = y0
            elif path[i] - path[i+1] >= self.N:
                x1 = x0
                y1 = y0 - 1
            elif path[i] - path[i+1] <= self.N:
                x1 = x0
                y1 = y0 + 1
            plt.plot([x0, x1], [y0, y1], color="red")

        x_start = np.mod(start, self.N)
        y_start = (start - x_start)/self.N
        x_end = np.mod(end, self.N)
        y_end = (end - x_end)/self.N
        plt.plot([x_start], [y_start], color="green", marker="o")
        plt.plot([x_end], [y_end], color="red", marker="o", zorder=3)
        plt.show()

        return np.array(path[::-1])

    # TODO: finish this!
    def random_walk(self, steps, start, end=None):
        """ compute random walk through the maze """

        agent = start
        traj = []
        for s in range(steps):
            if end is not None:
                if start == end:
                    return traj
            u = np.random.rand()
            col = self.T[agent, :].nonzero()[1]
            cmf = self.T[agent, col[0]]
            for i in range(1, len(col)):
                if u < cmf:
                    agent = col[candidates[i]]
                    traj.append(agent)
                    break

                cdf += self.T[agent, candidates[i]]

        return traj

    def visualize(self, wait=False, labels=False, committor=None):
        """ plot the maze """

        fig = plt.figure(figsize=(15, 10))

        for wall in self.walls_horizontal:
            x0 = np.mod(wall, self.N) + 0.5
            y0 = (wall-x0+0.5)/self.N - 0.5
            x1 = x0
            y1 = y0 + 1
            if labels:
                plt.text(x0, y0+0.5, str(wall))
            plt.plot([x0, x1], [y0, y1], color="black", zorder=2)

        for wall in self.walls_vertical:
            x0 = np.mod(wall, self.N) - 0.5
            y0 = (wall-x0-0.5)/self.N + 0.5 #+ 1
            x1 = x0 + 1
            y1 = y0
            if labels:
                plt.text(x0+0.5, y0, str(wall))
            plt.plot([x0, x1], [y0, y1], color="black", zorder=2)

        plt.plot([-0.5, -0.5], [-0.5, self.N-0.5], color="black", linewidth=2.0)
        plt.plot([-0.5, self.N-0.5], [-0.5, -0.5], color="black", linewidth=2.0)
        plt.plot([-0.5, self.N-0.5], [self.N-0.5, self.N-0.5], color="black", linewidth=2.0)
        plt.plot([self.N-0.5, self.N-0.5], [-0.5, self.N-0.5], color="black", linewidth=2.0)
        plt.xlim(-0.7, self.N-0.3)
        plt.ylim(-0.7, self.N-0.3)
        plt.xlabel("x")
        plt.ylabel("y")

        if labels:
            print("Walls:")
            print("vertical")
            print(self.walls_vertical)
            print("horizontal")
            print(self.walls_horizontal)
            print("sets:")
            print(self.__sets)

        fig.axes[0].set_aspect("equal")
        if committor is not None:
            com = np.zeros([self.N, self.N])
            x = np.zeros([self.N+1, self.N+1])
            y = np.zeros([self.N+1, self.N+1])
            for i in range(committor.shape[0]):
                x0 = np.mod(i, self.N)
                y0 = int((i-x0)/self.N)
                com[x0, y0] = committor[i]
                x[x0, y0] = float(x0) - 0.5
                y[x0, y0] = float(y0) - 0.5

            x[self.N, :] = x[:self.N, :self.N].max()+1
            y[:, self.N] = y[:self.N, :self.N].max()+1
            x[:, self.N] = x[:, self.N-1]
            y[self.N, :] = y[self.N-1, :]
            plt.pcolor(x, y, com, cmap="plasma")
            plt.colorbar()

        if not wait:
            plt.show()

    # method for recursive tree search
    def __path_search(self, row, col, start, end, parent):

        if start == end:
            return [start]
        if len(col[np.where(row == start)[0]]) < 1:
            return -1

        for c in col[np.where(row == start)[0]]:
            # don't go backwards
            if c != parent:
                # find path to end from child
                p = self.__path_search(row, col, c, end, start)
                if p != -1:
                    p.append(start)
                    return p
        # no path found
        return -1


def removearray(L, arr):
    """ function to remove a list from a list"""

    ind = 0
    size = len(L)
    while ind != size and not np.array_equal(L[ind], arr):
        ind += 1
    if ind != size:
        L.pop(ind)
    else:
        raise ValueError('array not found in list.')
