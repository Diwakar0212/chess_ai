import streamlit as st
import chess
import chess.svg
import base64
import json
import os
import time
from chess_engine import ChessEngine1500
from coach import ChessCoach

# --- PAGE CONFIG ---
st.set_page_config(page_title="Chess AI Coach", layout="wide")

# Shared state file path
STATE_FILE = os.path.join(os.path.dirname(__file__), "game_state.json")

# --- INITIALIZE STATE ---
if "board" not in st.session_state:
    st.session_state.board = chess.Board()
if "engine" not in st.session_state:
    st.session_state.engine = ChessEngine1500(max_depth=3)
if "coach" not in st.session_state:
    # Initialize Coach (Using Ollama by default)
    st.session_state.coach = ChessCoach(provider="ollama", model_name="llama3.2")
if "history" not in st.session_state:
    st.session_state.history = []
if "coach_explanation" not in st.session_state:
    st.session_state.coach_explanation = ""
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "last_player" not in st.session_state:
    st.session_state.last_player = None
if "last_state_ts" not in st.session_state:
    st.session_state.last_state_ts = 0

def sync_from_http_server():
    """Check if the HTTP server has pushed a new game state."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                state = json.load(f)
            ts = state.get("timestamp", 0)
            if ts > st.session_state.last_state_ts:
                st.session_state.last_state_ts = ts
                st.session_state.board = chess.Board(state["fen"])
                st.session_state.history = state.get("history", [])
                st.session_state.coach_explanation = state.get("coach_explanation", "")
                st.session_state.last_player = state.get("last_player")
                st.session_state.chat_messages = []
                # Hydrate coach context so follow-up questions work
                last_move = state.get("last_move", "")
                last_score = state.get("last_score", 0)
                last_player = state.get("last_player", "AI")
                if last_move:
                    st.session_state.coach.explain_move(
                        state["fen"], last_move, last_score, player=last_player
                    )
                return True
        except (json.JSONDecodeError, KeyError):
            pass
    return False

def save_state_for_http():
    """Write current state so the HTTP server stays in sync."""
    state = {
        "fen": st.session_state.board.fen(),
        "history": st.session_state.history,
        "coach_explanation": st.session_state.coach_explanation,
        "last_player": st.session_state.last_player,
        "timestamp": time.time()
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)
    st.session_state.last_state_ts = state["timestamp"]

# --- SYNC FROM HTTP SERVER ---
if sync_from_http_server():
    st.rerun()

# --- HELPER FUNCTIONS ---
def render_board(board):
    svg = chess.svg.board(board=board, size=400)
    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    return f'<img src="data:image/svg+xml;base64,{b64}" width="400" />'

# --- UI LAYOUT ---
st.title("♟️ Chess AI with LangChain Coach")

col_game, col_info = st.columns([1, 1])

with col_game:
    st.markdown(render_board(st.session_state.board), unsafe_allow_html=True)
    
    with st.form("move_form"):
        move_input = st.text_input("Your Move (e.g., e4, Nf3):")
        submitted = st.form_submit_button("Make Move")
        
    if submitted:
        board = st.session_state.board
        try:
            # 1. Player Move
            user_move = board.parse_san(move_input)
            if user_move in board.legal_moves:
                player_move_san = board.san(user_move)
                board.push(user_move)
                st.session_state.history.append(f"👤 You: {player_move_san}")
                
                # Get evaluation of player's move
                _, player_analysis = st.session_state.engine.get_best_move(board)
                
                # Coach explains YOUR move
                with st.spinner("Coach is analyzing your move..."):
                    player_explanation = st.session_state.coach.explain_move(
                        board.fen(), player_move_san, player_analysis['eval'], player="You"
                    )
                    st.session_state.coach_explanation = f"**Your move: {player_move_san}**\n\n{player_explanation}"
                    st.session_state.last_player = "You"
                    st.session_state.chat_messages = []
                
                if not board.is_game_over():
                    # 2. Engine Move
                    with st.spinner("Engine is calculating..."):
                        best_move, analysis = st.session_state.engine.get_best_move(board)
                        
                        # 3. Coach Explanation for AI move
                        with st.spinner("Coach is analyzing AI move..."):
                            explanation = st.session_state.coach.explain_move(
                                board.fen(), analysis['move'], analysis['eval'], player="AI"
                            )
                            st.session_state.coach_explanation = f"**AI's move: {analysis['move']}**\n\n{explanation}"
                            st.session_state.last_player = "AI"
                            st.session_state.chat_messages = []
                        
                        board.push(best_move)
                        st.session_state.history.append(f"🤖 AI ({analysis['move']}): {explanation}")
                        save_state_for_http()
                        st.rerun()
                else:
                    st.success("Game Over!")
            else:
                st.error("Illegal move!")
        except ValueError:
            st.error("Invalid move format.")

    if st.button("Reset Game"):
        st.session_state.board.reset()
        st.session_state.history = []
        st.session_state.coach_explanation = ""
        st.session_state.chat_messages = []
        st.session_state.last_player = None
        st.session_state.coach.clear_history()
        save_state_for_http()
        st.rerun()

with col_info:
    st.subheader("🎙️ Commentary")
    container = st.container(height=300)
    for msg in reversed(st.session_state.history):
        if "👤" in msg:
            container.markdown(f"**{msg}**")
        else:
            container.info(msg)
    
    # Cross-questioning section
    st.markdown("---")
    st.subheader("💬 Ask the Coach")
    st.caption("Ask about any move - yours or the AI's!")
    
    if st.session_state.coach_explanation:
        # Display chat history
        chat_container = st.container(height=200)
        for msg in st.session_state.chat_messages:
            if msg["role"] == "user":
                chat_container.markdown(f"**You:** {msg['content']}")
            else:
                chat_container.info(f"**Coach:** {msg['content']}")
        
        # Question input
        user_question = st.text_input("Ask about the last move or position:", key="follow_up_question", placeholder="e.g., What if I played differently?")
        
        col_q1, col_q2 = st.columns([3, 1])
        with col_q1:
            if st.button("Ask", use_container_width=True):
                if user_question:
                    with st.spinner("Coach is thinking..."):
                        answer = st.session_state.coach.ask_followup(user_question)
                        st.session_state.chat_messages.append({"role": "user", "content": user_question})
                        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
                        st.rerun()
        
        with col_q2:
            if st.button("Clear", use_container_width=True):
                st.session_state.chat_messages = []
                st.session_state.coach.clear_history()
                st.rerun()
    else:
        st.info("Make a move to start asking questions!")

# --- ASYNC POLL: check for HTTP server updates without reloading ---
@st.fragment(run_every=2)
def poll_http_state():
    """Background polling for HTTP server state changes."""
    if sync_from_http_server():
        st.rerun()

poll_http_state()