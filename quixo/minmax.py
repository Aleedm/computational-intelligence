from game import Player, Move
from game_wrapper import GameWrapper as Game
import random
from utilities import get_all_move, get_possible_moves, calculate_depth
from copy import deepcopy
from tqdm import tqdm


class MinMaxPlayer(Player):
    def __init__(self, max_depth=3, debug=False):
        super().__init__()
        self.player = None
        self.max_depth = max_depth
        self.debug = debug

    def make_move(self, game: "Game") -> tuple[tuple[int, int], Move]:
        best_score = float("-inf")
        best_move = None
        current_player = game.get_current_player()
        self.player = current_player
        with tqdm(
            get_possible_moves(game.get_board(), current_player, random=True),
            unit=" move",
            disable=not self.debug,
        ) as moves:
            depth = calculate_depth(len(moves), max_depth=self.max_depth)
            for move in moves:
                moves.set_description("Analyzing moves")

                board = deepcopy(game.get_board())
                cloned_game = Game(board, game.get_current_player())
                cloned_game.move(move[0], move[1], current_player)
                score = self.minimax(
                    cloned_game, depth, float("-inf"), float("inf"), False
                )
                if score > best_score:
                    best_score = score
                    best_move = move

                moves.set_postfix(
                    {
                        "current_move": move,
                        "best_move": best_move,
                        "best_score": best_score,
                    }
                )
        return best_move if best_move else random.choice(get_all_move())

    def minimax(self, game, depth, alpha, beta, is_maximizing):
        # print(f"deth: {depth}, alpha: {alpha}, beta: {beta}, is_maximizing: {is_maximizing}")
        current_player = game.get_current_player()
        if depth == 0 or game.check_winner() != -1:
            return self.evaluate_game(game)

        if is_maximizing:
            max_eval = float("-inf")
            for move in get_possible_moves(game.get_board(), current_player, random=True):
                cloned_game = game.clone()
                cloned_game.move(move[0], move[1], current_player)
                evaluation = self.minimax(cloned_game, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, evaluation)
                alpha = max(alpha, evaluation)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float("inf")
            for move in get_possible_moves(game.get_board(), current_player, random=True):
                cloned_game = game.clone()
                cloned_game.move(move[0], move[1], current_player)
                evaluation = self.minimax(cloned_game, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, evaluation)
                beta = min(beta, evaluation)
                if beta <= alpha:
                    break
            return min_eval

    def evaluate_game(self, game):
        winner = game.check_winner()
        if winner == -1:
            return 0
        elif winner == self.player:
            return 1
        else:
            return -1
