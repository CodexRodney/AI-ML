import sys

class Node():
    def __init__(self, state, parent, action):
        self.state = state
        self.parent = parent
        self.action = action

class StackFrontier:
    def __init__(self):
        self.frontier = []

    def add(self, node):
        self.frontier.append(node)

    def contains_state(self, state):
        """
        Returns True if it contains a node
        """
        return any(node.state == state for node in self.frontier)

    def empty(self):
        """
        To check if the frontier is empty
        """
        return len(self.frontier) == 0

    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            node = self.frontier[-1]
            self.frontier = self.frontier[:-1]
            return node

class QueueFrontier(StackFrontier):
    def remove(self):
        """
        Overrides the remove function to make it be like a queue
        """
        if self.empty():
            raise Exception("empty frontier")
        else:
            node = self.frontier[0]
            self.frontier = self.frontier[1:]
            return node
        

class Maze():
    def __init__(self, filename):
        # Read file and set height and width of maze
        with open(filename) as f:
            contents = f.read() # returns contents of file as a string

        # validate start and goal
        if contents.count("A") != 1:
            raise Exception("maze must have exactly one start point")
        if contents.count("B") != 1:
            raise Exception("maze must have exactly one goal")
        
        #Determine height and width of maze
        contents = contents.splitlines() # returns file contents as a list, each item is a list
        self.height = len(contents)
        self.width = max(len(line) for line in contents)

        #keep track of walls
        self.walls = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                try:
                    if contents[i][j] == "A":
                        self.start = (i, j)
                        row.apend(False)
                    elif contents[i][j] == "B":
                        self.goal = (i, j)
                        row.append(False)
                    elif contents[i][j] == " ":
                        row.append(False)
                    else:
                        row.append(False)
                except:
                    row.append(False)
            self.walls.append(row)

        self.solution = None