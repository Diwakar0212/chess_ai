"""
Simple HTTP server to handle POST requests for chess AI
No external framework required - uses built-in Python modules
"""
import json
import os
import time
import threading
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

def render_commentary_html(history):
    """Render commentary history matching Streamlit's display format."""
    html = '<div style="height: 300px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; background: #f9f9f9;">'
    for msg in reversed(history):
        # Check for player move (starts with 👤)
        if msg.startswith("👤"):
            # Player moves - bold markdown style with blue background (like Streamlit markdown)
            html += f'<div style="background: #f0f7ff; padding: 10px; margin: 8px 0; border-left: 3px solid #1f77b4; font-weight: bold; border-radius: 4px;">{msg}</div>'
        elif msg.startswith("🤖"):
            # AI moves - info box style (like Streamlit .info())
            html += f'<div style="background: #e8f4f8; padding: 10px; margin: 8px 0; border-left: 4px solid #0066cc; border-radius: 4px;">{msg}</div>'
        else:
            # Fallback - treat as AI move
            html += f'<div style="background: #e8f4f8; padding: 10px; margin: 8px 0; border-left: 4px solid #0066cc; border-radius: 4px;">{msg}</div>'
    html += '</div>'
    return html

def render_board_svg():
    """Render the current board as SVG."""
    return chess.svg.board(board=game_board, size=400)

# Persistent board to track game across requests
game_board = chess.Board()
game_history = []

# Initialize game state file
def initialize_game_state():
    """Initialize game state file if it doesn't exist."""
    if not os.path.exists(STATE_FILE):
        save_game_state(game_board.fen(), game_history, "", None)

initialize_game_state()

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
        global game_board, game_history
        
        if self.path == '/':
            self._set_headers()
            response = {
                "message": "Chess AI HTTP Server",
                "endpoints": {
                    "/move": "POST - Get AI move and explanation",
                    "/ui": "GET - Web UI with board and commentary",
                    "/health": "GET - Health check",
                    "/reset": "GET - Reset game"
                }
            }
            self.wfile.write(json.dumps(response).encode())
        
        elif self.path == '/ui':
            # Serve HTML UI matching Streamlit's layout
            self._set_headers()
            board_svg = render_board_svg()
            commentary_html = render_commentary_html(game_history)
            
            html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>♟️ Chess AI with LangChain Coach</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif; background: #f5f5f5; padding: 20px; }}
        h1 {{ text-align: center; color: #333; margin-bottom: 30px; }}
        .container {{ display: flex; gap: 20px; max-width: 1200px; margin: 0 auto; }}
        .game-column {{ flex: 1; }}
        .info-column {{ flex: 1; }}
        .board-container {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .board-container svg {{ max-width: 100%; height: auto; }}
        .form-group {{ margin-top: 15px; }}
        input {{ padding: 10px; width: 100%; border: 1px solid #ddd; border-radius: 4px; font-size: 14px; }}
        button {{ padding: 10px 20px; background: #0066cc; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; margin-top: 10px; width: 100%; }}
        button:hover {{ background: #0052a3; }}
        .commentary-section {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); height: 400px; display: flex; flex-direction: column; }}
        .commentary-section h2 {{ margin-bottom: 10px; color: #333; font-size: 18px; }}
        #commentary {{ flex: 1; overflow-y: auto; border: 1px solid #ddd; padding: 10px; background: #fafafa; border-radius: 4px; }}
        .reset-button {{ background: #ff6b6b; margin-top: 10px; }}
        .reset-button:hover {{ background: #ee5a52; }}
        .status-msg {{ padding: 10px; margin-top: 10px; border-radius: 4px; text-align: center; font-weight: bold; }}
        .status-success {{ background: #d4edda; color: #155724; }}
        .status-loading {{ background: #cce5ff; color: #004085; }}
        .status-error {{ background: #f8d7da; color: #721c24; }}
    </style>
</head>
<body>
    <h1>♟️ Chess AI with LangChain Coach</h1>
    
    <div class="container">
        <div class="game-column">
            <div class="board-container">
                {board_svg}
                <div class="form-group">
                    <input type="text" id="moveInput" placeholder="Your Move (e.g., e4, Nf3)" autofocus>
                    <button onclick="makeMove()">Make Move</button>
                    <button class="reset-button" onclick="resetGame()">Reset Game</button>
                </div>
                <div id="status" class="status-msg"></div>
            </div>
        </div>
        
        <div class="info-column">
            <div class="commentary-section">
                <h2>🎙️ Commentary</h2>
                <div id="commentary">
                    {commentary_html}
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let moveInProgress = false;
        let lastCommentaryCount = {len(game_history)};
        
        function makeMove() {{
            if (moveInProgress) return;
            
            const move = document.getElementById('moveInput').value.trim();
            if (!move) {{
                showStatus('Please enter a move', 'error');
                return;
            }}
            
            moveInProgress = true;
            showStatus('Engine thinking...', 'loading');
            
            // STAGE 1: Send move to server
            fetch('/move', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify({{move: move}})
            }})
            .then(r => r.json())
            .then(data => {{
                if (data.error) {{
                    showStatus('Error: ' + data.error, 'error');
                    moveInProgress = false;
                    return;
                }}
                
                // STAGE 2: Board updates immediately
                showStatus('Board updated. Coach analyzing...', 'loading');
                document.getElementById('moveInput').value = '';
                
                // Reload page IMMEDIATELY to show board
                setTimeout(() => {{
                    location.reload();
                }}, 300);
                
                // STAGE 3: Start polling for commentary updates
                let pollCount = 0;
                const pollInterval = setInterval(() => {{
                    pollCount++;
                    if (pollCount > 30) {{ // Stop after 60 seconds
                        clearInterval(pollInterval);
                        return;
                    }}
                    
                    fetch('/commentary-status')
                        .then(r => r.json())
                        .then(response => {{
                            if (response.total_moves > lastCommentaryCount) {{
                                lastCommentaryCount = response.total_moves;
                                clearInterval(pollInterval);
                                
                                // Commentary is ready - reload page to show it
                                setTimeout(() => {{
                                    location.reload();
                                }}, 500);
                            }}
                        }})
                        .catch(e => console.log('Poll error:', e));
                }}, 2000); // Poll every 2 seconds
            }})
            .catch(e => {{
                showStatus('Error: ' + e.message, 'error');
                moveInProgress = false;
            }});
        }}
        
        function resetGame() {{
            if (!confirm('Reset game? All moves will be lost.')) return;
            
            fetch('/reset')
                .then(r => r.json())
                .then(() => {{
                    lastCommentaryCount = 0;
                    showStatus('Game reset!', 'success');
                    setTimeout(() => location.reload(), 300);
                }})
                .catch(e => showStatus('Error: ' + e.message, 'error'));
        }}
        
        function showStatus(msg, type) {{
            const statusDiv = document.getElementById('status');
            statusDiv.textContent = msg;
            statusDiv.className = 'status-msg status-' + type;
        }}
        
        document.getElementById('moveInput').addEventListener('keypress', (e) => {{
            if (e.key === 'Enter') makeMove();
        }});
    </script>
</body>
</html>
            """
            self.wfile.write(html.encode())
        
        elif self.path == '/commentary-status':
            # Return current commentary status for polling
            self._set_headers()
            self.wfile.write(json.dumps({
                "history": game_history,
                "fen": game_board.fen(),
                "total_moves": len(game_history)
            }).encode())
        
        elif self.path == '/reset':
            game_board.reset()
            game_history.clear()
            engine.reset_engine()
            coach.clear_history()
            save_game_state(game_board.fen(), game_history, "", None)
            self._set_headers()
            self.wfile.write(json.dumps({"status": "reset_complete"}).encode())
        
        elif self.path == '/health':
            self._set_headers()
            health_data = {
                "status": "healthy",
                "engine": "ChessEngine1500",
                "coach": "LangChain (Ollama)",
                "board_fen": game_board.fen(),
                "moves_made": len(game_history),
                "game_over": game_board.is_game_over()
            }
            self.wfile.write(json.dumps(health_data).encode())
        
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
            
            # --- STAGE 1: PLAYER MOVE ---
            player_move_san = game_board.san(move)
            game_board.push(move)
            
            # Evaluate player's move
            _, player_analysis = engine.get_best_move(game_board)
            
            # Coach explains YOUR move (BLOCKING - must complete)
            player_explanation = coach.explain_move(
                game_board.fen(), player_move_san,
                player_analysis['eval'], player="You"
            )
            
            # Add to history
            game_history.append(f"👤 You ({player_move_san}): {player_explanation}")
            
            # --- STAGE 2: ENGINE MOVE ---
            ai_move_san = ""
            coach_explanation = f"**Your move: {player_move_san}**\n\n{player_explanation}"
            last_move = player_move_san
            last_score = player_analysis['eval']
            last_player = "You"
            
            if not game_board.is_game_over():
                best_move, analysis = engine.get_best_move(game_board)
                if best_move:
                    ai_move_san = game_board.san(best_move)
                    game_board.push(best_move)
                    
                    # Add placeholder for AI move (will be updated by async thread)
                    game_history.append(f"🤖 AI ({ai_move_san}): [Coach analyzing...]")
                    
                    last_move = ai_move_san
                    last_score = analysis['eval']
                    last_player = "AI"
                    
                    # Save state IMMEDIATELY with updated board (matches Streamlit rerun)
                    save_game_state(
                        game_board.fen(), game_history,
                        coach_explanation, last_player, last_move, last_score
                    )
                    
                    # IMPORTANT: Start async coach explanation in background thread
                    # DON'T WAIT - return response immediately
                    def explain_engine_move_async():
                        try:
                            ai_explanation = coach.explain_move(
                                game_board.fen(), ai_move_san,
                                analysis['eval'], player="AI"
                            )
                            # Update history with actual explanation
                            game_history[-1] = f"🤖 AI ({ai_move_san}): {ai_explanation}"
                            # Update saved state with full explanation
                            save_game_state(
                                game_board.fen(), game_history,
                                f"**AI's move: {ai_move_san}**\n\n{ai_explanation}",
                                last_player, last_move, last_score
                            )
                        except Exception as e:
                            print(f"Coach commentary error: {e}")
                    
                    thread = threading.Thread(target=explain_engine_move_async, daemon=True)
                    thread.start()
                    
                    # --- SEND RESPONSE IMMEDIATELY (matches Streamlit rerun) ---
                    self._set_headers(200)
                    self.wfile.write(json.dumps({
                        "move": ai_move_san,
                        "eval": last_score,
                        "board_fen": game_board.fen(),
                        "status": "move_made_commentary_loading"
                    }).encode())
                    return
            
            # Game over case
            save_game_state(game_board.fen(), game_history, coach_explanation, last_player, last_move, last_score)
            
            self._set_headers(200)
            self.wfile.write(json.dumps({
                "move": ai_move_san,
                "board_fen": game_board.fen(),
                "game_over": True,
                "status": "game_over"
            }).encode())
        
        except Exception as e:
            self._set_headers(500)
            self.wfile.write(json.dumps({"error": str(e)}).encode())
        
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
