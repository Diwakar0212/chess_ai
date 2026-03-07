"""
Simple HTTP server to handle POST requests for chess AI
No external framework required - uses built-in Python modules
"""
import json
import os
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import chess
from chess_engine import ChessEngine1500
from coach import ChessCoach

# Shared state file path
STATE_FILE = os.path.join(os.path.dirname(__file__), "game_state.json")

# Initialize engine and coach
engine = ChessEngine1500(max_depth=3)
coach = ChessCoach(provider="ollama", model_name="llama3.2")

def save_game_state(board_fen, history, coach_explanation, last_player, last_move="", last_score=0):
    """Write game state to shared file so Streamlit can pick it up."""
    state = {
        "fen": board_fen,
        "history": history,
        "coach_explanation": coach_explanation,
        "last_player": last_player,
        "last_move": last_move,
        "last_score": last_score,
        "timestamp": time.time()
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

def load_game_state():
    """Load current game state from shared file."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return None

# Persistent board to track game across requests
game_board = chess.Board()
game_history = []

class ChessRequestHandler(BaseHTTPRequestHandler):
    
    def _set_headers(self, status=200):
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_OPTIONS(self):
        self._set_headers()
    
    def do_GET(self):
        if self.path == '/':
            self._set_headers()
            response = {
                "message": "Chess AI HTTP Server",
                "endpoints": {
                    "/move": "POST - Get AI move and explanation",
                    "/explain": "POST - Explain a specific move",
                    "/health": "GET - Health check"
                }
            }
            self.wfile.write(json.dumps(response).encode())
        
        elif self.path == '/reset':
            game_board.reset()
            game_history.clear()
            save_game_state(game_board.fen(), game_history, "", None)
            self._set_headers()
            self.wfile.write(json.dumps({"move": "reset"}).encode())
        
        elif self.path == '/health':
            self._set_headers()
            self.wfile.write(json.dumps({"status": "healthy"}).encode())
        
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Not found"}).encode())
    
    def do_POST(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
        except (json.JSONDecodeError, ValueError):
            self._set_headers(400)
            self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode())
            return

        if self.path == '/move':
            self._handle_move(request_data)
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode())
    
    def _handle_move(self, data):
        global game_board, game_history
        try:
            player_move = data.get('move', None)
            
            if not player_move:
                self._set_headers(400)
                self.wfile.write(json.dumps({"move": "error"}).encode())
                return
            
            # Parse SAN move (e.g. "e4", "Nf3")
            try:
                move = game_board.parse_san(player_move)
            except ValueError:
                self._set_headers(400)
                self.wfile.write(json.dumps({"move": "illegal"}).encode())
                return
            
            if move not in game_board.legal_moves:
                self._set_headers(400)
                self.wfile.write(json.dumps({"move": "illegal"}).encode())
                return
            
            # Apply player move
            player_move_san = game_board.san(move)
            game_board.push(move)
            game_history.append(f"\ud83d\udc64 You: {player_move_san}")
            
            # Evaluate and explain player's move
            _, player_analysis = engine.get_best_move(game_board)
            player_explanation = coach.explain_move(
                game_board.fen(), player_move_san,
                player_analysis['eval'], player="You"
            )
            
            # Get AI move
            ai_move_san = ""
            ai_explanation = ""
            coach_explanation = f"**Your move: {player_move_san}**\n\n{player_explanation}"
            last_move = player_move_san
            last_score = player_analysis['eval']
            last_player = "You"
            
            if not game_board.is_game_over():
                best_move, analysis = engine.get_best_move(game_board)
                if best_move:
                    ai_move_san = game_board.san(best_move)
                    
                    ai_explanation = coach.explain_move(
                        game_board.fen(), ai_move_san,
                        analysis['eval'], player="AI"
                    )
                    
                    game_board.push(best_move)
                    game_history.append(f"\ud83e\udd16 AI ({ai_move_san}): {ai_explanation}")
                    coach_explanation = f"**Your move: {player_move_san}**\n\n{player_explanation}\n\n---\n\n**AI's move: {ai_move_san}**\n\n{ai_explanation}"
                    last_move = ai_move_san
                    last_score = analysis['eval']
                    last_player = "AI"
            
            # Save state for Streamlit
            save_game_state(game_board.fen(), game_history, coach_explanation, last_player, last_move, last_score)
            
            # Response: just the AI move
            self._set_headers(200)
            self.wfile.write(json.dumps({"move": ai_move_san}).encode())
        
        except Exception as e:
            self._set_headers(500)
            self.wfile.write(json.dumps({"error": str(e)}).encode())
    
    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {format % args}")

def run_server(host='0.0.0.0', port=8000):
    server = HTTPServer((host, port), ChessRequestHandler)
    print(f"Chess AI HTTP Server running on http://{host}:{port}")
    print(f"Press Ctrl+C to stop\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")
        server.server_close()

if __name__ == "__main__":
    run_server()
