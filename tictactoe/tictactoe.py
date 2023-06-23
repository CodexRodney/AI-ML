"""
Tic Tac Toe Player
"""

import math
import copy
X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]

def player(board):
    """
    Returns player who has the next turn on a board.
    """
    # checks if game is still on
    if terminal(board) == True:
        return "Game is Over"
      
    if (len(actions(board)) % 2) == 0:
        return O
    return X

def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    # checks if the game is still on
    if terminal(board):
        return "Game is over"
    
    possible_actions = set()
    start = 0
    for x in board:
        index = 0
        for w in x:
            if EMPTY == w:
                possible_actions.add((start, index))
            index += 1
        start += 1
    return possible_actions

def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    possible_actions = actions(board)
    if isinstance(possible_actions, set):
        if action in possible_actions:
            # deep copy of original board
            new_board = copy.deepcopy(board)
            # inserts the next player in the action set
            new_board[action[0]][action[1]] = player(new_board)
            return new_board
        else:
            raise ValueError("Inappropriate Move")


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # check if O has won the game
    if (horizontal_winning(board, O) or vertical_winning(board, O)
            or diagonal_win(board, O)):
        return O
    # check if X has won the game
    if (horizontal_winning(board, X) or vertical_winning(board, X)
            or diagonal_win(board, X)):
        return X
    return None

def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    # checks if the board has places that are not filled
    if not any(EMPTY in x for x in board):
        return True
    if winner(board) is not None:
        return True
    return False

def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    else:
        return 0

def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    #Assuming the role of a minimal player without playing optimally
    if terminal(board):
        return None
    print("The next player is ", player(board))
    if player(board) == O:
        move = min_player(board)[0]
        print("The move is ", move)
        return move
    elif player(board) == X:
        move = max_player(board)[0]
        print("The move is ", move)
        return move
    # new_board = copy.deepcopy(board)
    # value = -math.inf
    # best_action = None
    # for action in actions(new_board):
    #     # ai to play move
    #     new_board = result(new_board, action)
    #     oponent_new_board = copy.deepcopy(new_board)
    #     # taking account of the ai's move
    #     for oponent_action in actions(oponent_new_board):
    #         # taking into consideration oponent's move
    #         print("The possible actions are: ", actions(oponent_new_board))
    #         oponent_new_board = result(oponent_new_board, oponent_action)
    #         utilize = utility(oponent_new_board)
    #         if value < utilize:
    #             # print("The Action: ", oponent_action)
    #             # print("The value is : ", value)
    #             best_action = oponent_action
    #             value = utilize
    #         oponent_new_board = new_board
    #     new_board = board
    # return best_action
    
def max_player(board):
    # checks if the game is at the end
    if terminal(board):
        return utility(board)
    
    print("The board in max_player:")
    print(board)
    v = -math.inf # will hold the utility with the highest value
    best_action = None # plays the best move to be
    for action in actions(board):
        print("Considering another move in the max player")
        current_utility = min_player(result(board, action))
        print(f"In the max player the utility is {current_utility}")
        if isinstance(current_utility, tuple):
            if current_utility[1] > v:
                v = current_utility[1]
                best_action = action
        elif current_utility > v:
            v = current_utility
            best_action = action
    return (best_action, v)
        

def min_player(board):
    if terminal(board):
        return utility(board)
    
    print("The board in min_player:")
    print(board)
    v = math.inf
    best_action = None
    for action in actions(board):
        current_utility = max_player(result(board, action))
        print(f"In the min player the utility is {current_utility}")
        if isinstance(current_utility, tuple):
            if current_utility[1] < v:
                v = current_utility[1]
                best_action = action
        elif current_utility < v:
            v = current_utility
            best_action = action
    return (best_action, v)

def diagonal_win(board, character):
    """
    Returns True if character occupies all
    diagonal positions in the board provided
    """
    diagonal_indices = [(0,0), (1,1), (2,2)]
    diagonal_board = [board[x[0]][x[1]] for x in diagonal_indices]
    count = 0
    for x in diagonal_board:
        if character == x:
            count += 1
    if count == 3:
        return True
    # checking for the other diagonal
    count = 0
    diagonal_indices= [(0,2), (1,1), (2,0)]
    diagonal_board = [board[x[0]][x[1]] for x in diagonal_indices]
    for x in diagonal_board:
        if character == x:
            count += 1
    if count == 3:
        return True

    return False

def horizontal_winning(board, character):
    """
    Returns True if character occupies any of 
    all the horizontal positions of board
    """
    # print("The board being checked is\n", board)
    # print("The Character being checked is ", character)
    for x in board:
        count = 0
        for w in x:
            if character == w:
                count += 1
        if count == 3:
            return True
    return False

def vertical_winning(board, character):
    """
    Returns True if character occupies any of
    all the vertical positions of the board"""
    for x in zip(*board):
        count = 0
        for w in x:
            if character == w:
                count += 1
        if count == 3:
            return True
    return False

