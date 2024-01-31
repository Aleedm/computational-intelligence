from game import Player, Move
from game_wrapper import GameWrapper as Game
import random
from utilities import get_possible_moves, calculate_depth
from copy import deepcopy
from tqdm import tqdm

# Define a MinMaxPlayer class, inheriting from Player
class MinMaxPlayer(Player):
    def __init__(self, max_depth=3, debug=False):
        # Initialize the player with maximum search depth and debug mode
        super().__init__()
        self.player = None
        self.max_depth = max_depth
        self.debug = debug

    def make_move(self, game: "Game") -> tuple[tuple[int, int], Move]:
        # Method to make a move in the game
        best_score = float("-inf")
        best_move = None
        current_player = game.get_current_player()
        self.player = current_player

        # Iterate through all possible moves, with a progress bar if in debug mode
        with tqdm(
            get_possible_moves(game.get_board(), current_player, random=True),
            unit=" move",
            disable=not self.debug,
        ) as moves:
            depth = calculate_depth(len(moves), max_depth=self.max_depth)
            for move in moves:
                moves.set_description("Analyzing moves")

                # Clone the game board and make the move
                board = deepcopy(game.get_board())
                cloned_game = Game(board, game.get_current_player())
                cloned_game.move(move[0], move[1], current_player)

                # Use minimax to evaluate the move
                score = self.minimax(
                    cloned_game, depth, float("-inf"), float("inf"), False
                )
                if score > best_score:
                    best_score = score
                    best_move = move

                # Update progress bar with the current move, best move, and score
                moves.set_postfix(
                    {
                        "current_move": move,
                        "best_move": best_move,
                        "best_score": best_score,
                    }
                )
        # Return the best move found, or a random move if no best move is determined
        if best_move:
            return best_move
        else: 
            return random.choice(get_possible_moves(game.get_board(), current_player, random=True)), get_possible_moves(game.get_board(), current_player, random=False)

    def minimax(self, game, depth, alpha, beta, is_maximizing):
        # Minimax algorithm with alpha-beta pruning
        current_player = game.get_current_player()
        if depth == 0 or game.check_winner() != -1:
            # Evaluate the game if it's at max depth or a winner is found
            return self.evaluate_game(game)

        if is_maximizing:
            max_eval = float("-inf")
            for move in get_possible_moves(game.get_board(), current_player, random=True):
                # Clone the game and make the move
                cloned_game = game.clone()
                cloned_game.move(move[0], move[1], current_player)

                # Recursive call to minimax
                evaluation = self.minimax(cloned_game, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, evaluation)
                alpha = max(alpha, evaluation)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float("inf")
            for move in get_possible_moves(game.get_board(), current_player, random=True):
                # Similar to the maximizing player, but minimizing the score
                cloned_game = game.clone()
                cloned_game.move(move[0], move[1], current_player)
                evaluation = self.minimax(cloned_game, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, evaluation)
                beta = min(beta, evaluation)
                if beta <= alpha:
                    break
            return min_eval

    def evaluate_game(self, game):
        # Evaluate the game state
        winner = game.check_winner()
        # Return 0 for a tie, 1 for a win, and -1 for a loss
        if winner == -1:
            return 0
        elif winner == self.player:
            return 1
        else:
            return -1
