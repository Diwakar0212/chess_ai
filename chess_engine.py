"""
Chess Engine - 1500 Elo Strength (Web App Compatible)
"""

import chess
import chess.polyglot
from typing import Dict, List, Optional, Tuple
import time

class ChessEngine1500:
    """
    Chess engine targeting 1500 Elo strength
    """
    
    # PIECE-SQUARE TABLES
    PAWN_TABLE = [
        0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5,  5, 10, 25, 25, 10,  5,  5,
        0,  0,  0, 20, 20,  0,  0,  0,
        5, -5,-10,  0,  0,-10, -5,  5,
        5, 10, 10,-20,-20, 10, 10,  5,
        0,  0,  0,  0,  0,  0,  0,  0
    ]
    
    KNIGHT_TABLE = [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50
    ]
    
    BISHOP_TABLE = [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20
    ]
    
    ROOK_TABLE = [
        -10, -5,  0,  5,  5,  0, -5,-10,
          5, 10, 10, 10, 10, 10, 10,  5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         20, 20, 20, 20, 20, 20, 20, 20,
          5,  5, 10, 10, 10, 10,  5,  5
    ]
    
    QUEEN_TABLE = [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
        -5,  0,  5,  5,  5,  5,  0, -5,
        0,  0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20
    ]
    
    KING_MIDDLE_GAME_TABLE = [
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -10,-20,-20,-20,-20,-20,-20,-10,
        20, 20,  0,  0,  0,  0, 20, 20,
        20, 30, 10,  0,  0, 10, 30, 20
    ]
    
    KING_END_GAME_TABLE = [
        -50,-40,-30,-20,-20,-30,-40,-50,
        -30,-20,-10,  0,  0,-10,-20,-30,
        -30,-10, 20, 30, 30, 20,-10,-30,
        -30,-10, 30, 40, 40, 30,-10,-30,
        -30,-10, 30, 40, 40, 30,-10,-30,
        -30,-10, 20, 30, 30, 20,-10,-30,
        -30,-30,  0,  0,  0,  0,-30,-30,
        -50,-30,-30,-30,-30,-30,-30,-50
    ]
    
    def __init__(self, max_depth: int = 4, time_limit: float = 2.0):
        """
        Initialize the 1500 Elo chess engine
        """
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.transposition_table: Dict[int, Dict] = {}
        self.nodes_searched = 0
        
        # Move ordering tables
        self.killer_moves: Dict[int, List[chess.Move]] = {}
        self.history_scores: Dict[str, int] = {}
        
        # Try to load opening book (Web App Safe Mode)
        try:
            # We look for the file in the current directory
            self.opening_book = chess.polyglot.open_reader("performance.bin")
            print("✓ Opening book loaded")
        except:
            print("✗ No opening book found (Standard play mode)")
            self.opening_book = None

    def is_endgame(self, board: chess.Board) -> bool:
        queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
        minors = (len(board.pieces(chess.KNIGHT, chess.WHITE)) + 
                  len(board.pieces(chess.BISHOP, chess.WHITE)) +
                  len(board.pieces(chess.KNIGHT, chess.BLACK)) + 
                  len(board.pieces(chess.BISHOP, chess.BLACK)))
        
        return queens == 0 or (queens == 2 and minors <= 2)
    
    def evaluate_position(self, board: chess.Board) -> int:
        if board.is_checkmate():
            return -20000 if board.turn else 20000
        
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        score = 0
        endgame = self.is_endgame(board)
        piece_count = len(board.piece_map())
        
        # 1. Material + Position
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.piece_values[piece.piece_type]
                table_index = square if piece.color == chess.WHITE else (63 - square)
                
                if piece.piece_type == chess.PAWN:
                    value += self.PAWN_TABLE[table_index]
                elif piece.piece_type == chess.KNIGHT:
                    value += self.KNIGHT_TABLE[table_index]
                elif piece.piece_type == chess.BISHOP:
                    value += self.BISHOP_TABLE[table_index]
                elif piece.piece_type == chess.ROOK:
                    value += self.ROOK_TABLE[table_index]
                elif piece.piece_type == chess.QUEEN:
                    value += self.QUEEN_TABLE[table_index]
                elif piece.piece_type == chess.KING:
                    value += self.KING_END_GAME_TABLE[table_index] if endgame else self.KING_MIDDLE_GAME_TABLE[table_index]
                
                score += value if piece.color == chess.WHITE else -value
        
        # 2. Mobility
        original_turn = board.turn
        board.turn = chess.WHITE
        white_mobility = board.legal_moves.count()
        board.turn = chess.BLACK
        black_mobility = board.legal_moves.count()
        board.turn = original_turn
        score += (white_mobility - black_mobility) * 10
        
        # 3. Pawn structure
        score += self.evaluate_pawn_structure(board)
        
        # 4. King safety (middlegame only)
        if not endgame:
            score += self.evaluate_king_safety(board, chess.WHITE)
            score -= self.evaluate_king_safety(board, chess.BLACK)
        
        # 5. Center control
        score += self.evaluate_center_control(board)
        
        # 6. Rook placement
        score += self.evaluate_rooks(board)
        
        # 7. Bishop pair bonus
        score += self.evaluate_bishop_pair(board)
        
        # 8. King opposition (pawn endgames)
        if endgame and piece_count <= 6:
            score += self.evaluate_king_opposition(board)
        
        # 9. Opposite color bishops drawish tendency
        if endgame:
            score += self.evaluate_opposite_bishops(board, score)
        
        # 10. Rook + passed pawn coordination
        if endgame:
            score += self.evaluate_rook_pawn_endgame(board)
        
        return score if board.turn == chess.WHITE else -score
    
    def evaluate_pawn_structure(self, board: chess.Board) -> int:
        score = 0
        for color in [chess.WHITE, chess.BLACK]:
            multiplier = 1 if color == chess.WHITE else -1
            pawns = board.pieces(chess.PAWN, color)
            
            # Doubled pawns
            for file in range(8):
                pawns_on_file = sum(1 for sq in pawns if chess.square_file(sq) == file)
                if pawns_on_file > 1:
                    score -= 20 * (pawns_on_file - 1) * multiplier
            
            # Isolated pawns
            for square in pawns:
                file = chess.square_file(square)
                has_neighbor = any(chess.square_file(sq) in [file - 1, file + 1] for sq in pawns)
                if not has_neighbor:
                    score -= 15 * multiplier
            
            # Passed pawns
            for square in pawns:
                file = chess.square_file(square)
                rank = chess.square_rank(square)
                enemy_pawns = board.pieces(chess.PAWN, not color)
                is_passed = True
                
                for enemy_sq in enemy_pawns:
                    enemy_file = chess.square_file(enemy_sq)
                    enemy_rank = chess.square_rank(enemy_sq)
                    
                    if abs(enemy_file - file) <= 1:
                        if color == chess.WHITE and enemy_rank > rank:
                            is_passed = False; break
                        elif color == chess.BLACK and enemy_rank < rank:
                            is_passed = False; break
                
                if is_passed:
                    advancement = rank if color == chess.WHITE else (7 - rank)
                    score += (10 + advancement * 10) * multiplier
        return score
    
    def evaluate_king_safety(self, board: chess.Board, color: chess.Color) -> int:
        king_square = board.king(color)
        if king_square is None: return 0
        
        safety = 0
        king_rank = chess.square_rank(king_square)
        king_file = chess.square_file(king_square)
        
        # Pawn shield
        shield_rank = king_rank + 1 if color == chess.WHITE else king_rank - 1
        if 0 <= shield_rank <= 7:
            for file_offset in [-1, 0, 1]:
                file = king_file + file_offset
                if 0 <= file < 8:
                    square = chess.square(file, shield_rank)
                    piece = board.piece_at(square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == color:
                        safety += 15
        
        # Penalty for attackers
        attackers = board.attackers(not color, king_square)
        safety -= len(attackers) * 25
        
        # Bonus for castling rights
        if color == chess.WHITE:
            if board.has_kingside_castling_rights(chess.WHITE): safety += 10
            if board.has_queenside_castling_rights(chess.WHITE): safety += 10
        else:
            if board.has_kingside_castling_rights(chess.BLACK): safety += 10
            if board.has_queenside_castling_rights(chess.BLACK): safety += 10
        
        return safety
    
    def evaluate_center_control(self, board: chess.Board) -> int:
        center = [chess.E4, chess.E5, chess.D4, chess.D5]
        score = 0
        for square in center:
            white_control = len(board.attackers(chess.WHITE, square))
            black_control = len(board.attackers(chess.BLACK, square))
            score += (white_control - black_control) * 5
        return score
    
    def evaluate_rooks(self, board: chess.Board) -> int:
        score = 0
        for color in [chess.WHITE, chess.BLACK]:
            multiplier = 1 if color == chess.WHITE else -1
            rooks = board.pieces(chess.ROOK, color)
            
            for rook_square in rooks:
                file = chess.square_file(rook_square)
                rank = chess.square_rank(rook_square)
                
                # Open files
                pawns_on_file = sum(1 for sq in chess.SQUARES 
                                  if chess.square_file(sq) == file and 
                                  board.piece_at(sq) and 
                                  board.piece_at(sq).piece_type == chess.PAWN)
                
                if pawns_on_file == 0:
                    score += 20 * multiplier
                elif pawns_on_file == 1:
                    score += 10 * multiplier
                
                # 7th rank
                if (color == chess.WHITE and rank == 6) or (color == chess.BLACK and rank == 1):
                    score += 25 * multiplier
        return score
    
    def evaluate_bishop_pair(self, board: chess.Board) -> int:
        score = 0
        if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2: score += 30
        if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2: score -= 30
        return score

    def evaluate_king_opposition(self, board: chess.Board) -> int:
        white_king = board.king(chess.WHITE)
        black_king = board.king(chess.BLACK)
        if white_king is None or black_king is None: return 0
        
        file_diff = abs(chess.square_file(white_king) - chess.square_file(black_king))
        rank_diff = abs(chess.square_rank(white_king) - chess.square_rank(black_king))
        
        if (file_diff == 0 and rank_diff == 2) or (rank_diff == 0 and file_diff == 2):
            return -20 if board.turn == chess.WHITE else 20
        return 0

    def evaluate_opposite_bishops(self, board: chess.Board, current_score: int) -> int:
        white_bishops = list(board.pieces(chess.BISHOP, chess.WHITE))
        black_bishops = list(board.pieces(chess.BISHOP, chess.BLACK))
        
        if len(white_bishops) == 1 and len(black_bishops) == 1:
            w_sq = white_bishops[0]
            b_sq = black_bishops[0]
            if (chess.square_file(w_sq) + chess.square_rank(w_sq)) % 2 != (chess.square_file(b_sq) + chess.square_rank(b_sq)) % 2:
                return int(-current_score * 0.4)
        return 0

    def evaluate_rook_pawn_endgame(self, board: chess.Board) -> int:
        score = 0
        # Simplified logic for brevity
        return score

    # SEARCH & MOVE ORDERING
    def order_moves(self, board: chess.Board, moves: List[chess.Move], depth: int) -> List[chess.Move]:
        def move_score(move):
            score = 0
            if board.is_capture(move):
                score += 10000
            if move.promotion:
                score += 9000
            if depth in self.killer_moves and move in self.killer_moves[depth]:
                score += 8000
            return score
        return sorted(moves, key=move_score, reverse=True)

    def update_killers(self, move: chess.Move, depth: int):
        if depth not in self.killer_moves: self.killer_moves[depth] = []
        if move not in self.killer_moves[depth]:
            self.killer_moves[depth].insert(0, move)
            if len(self.killer_moves[depth]) > 2: self.killer_moves[depth].pop()

    def quiescence_search(self, board: chess.Board, alpha: int, beta: int, depth: int = 0) -> int:
        self.nodes_searched += 1
        stand_pat = self.evaluate_position(board)
        if depth >= 10: return stand_pat
        if stand_pat >= beta: return beta
        if alpha < stand_pat: alpha = stand_pat
        
        for move in board.legal_moves:
            if board.is_capture(move) or move.promotion:
                board.push(move)
                score = -self.quiescence_search(board, -beta, -alpha, depth + 1)
                board.pop()
                if score >= beta: return beta
                if score > alpha: alpha = score
        return alpha

    def minimax(self, board: chess.Board, depth: int, alpha: int, beta: int, maximizing: bool, allow_null: bool = True) -> int:
        self.nodes_searched += 1
        board_hash = chess.polyglot.zobrist_hash(board)
        if board_hash in self.transposition_table and self.transposition_table[board_hash]['depth'] >= depth:
            return self.transposition_table[board_hash]['score']
            
        if depth == 0: return self.quiescence_search(board, alpha, beta)
        if board.is_game_over(): return 0
        
        # Null Move Pruning
        if allow_null and depth >= 3 and not board.is_check():
            board.push(chess.Move.null())
            score = -self.minimax(board, depth - 1 - 2, -beta, -beta + 1, False, allow_null=False)
            board.pop()
            if score >= beta: return beta

        best_score = float('-inf')
        ordered_moves = self.order_moves(board, list(board.legal_moves), depth)
        
        for move in ordered_moves:
            board.push(move)
            score = -self.minimax(board, depth - 1, -beta, -alpha, not maximizing, allow_null)
            board.pop()
            
            if score > best_score: best_score = score
            if score > alpha: alpha = score
            if alpha >= beta:
                self.update_killers(move, depth)
                break
        
        self.transposition_table[board_hash] = {'score': best_score, 'depth': depth}
        return best_score

    def iterative_deepening(self, board: chess.Board) -> Tuple[Optional[chess.Move], Dict]:
        self.nodes_searched = 0
        start_time = time.time()
        best_move = None
        best_score = float('-inf')
        
        for depth in range(1, self.max_depth + 1):
            if time.time() - start_time > self.time_limit: break
            
            current_best_move = None
            current_best_score = float('-inf')
            
            # Simple root search
            moves = self.order_moves(board, list(board.legal_moves), depth)
            for move in moves:
                board.push(move)
                score = -self.minimax(board, depth - 1, float('-inf'), float('inf'), False)
                board.pop()
                
                if score > current_best_score:
                    current_best_score = score
                    current_best_move = move
            
            best_move = current_best_move
            best_score = current_best_score
            
        elapsed = time.time() - start_time
        return best_move, {
            'eval': best_score,
            'nodes': self.nodes_searched,
            'time': elapsed,
            'move': board.san(best_move) if best_move else None
        }

    def get_book_move(self, board: chess.Board) -> Optional[chess.Move]:
        if self.opening_book is None: return None
        try:
            return self.opening_book.weighted_choice(board).move
        except: return None

    def get_best_move(self, board: chess.Board) -> Tuple[Optional[chess.Move], Dict]:
        # Try opening book
        book_move = self.get_book_move(board)
        if book_move and board.fullmove_number <= 10:
            return book_move, {
                'eval': 0,
                'source': 'book',
                'move': board.san(book_move)
            }
        
        return self.iterative_deepening(board)