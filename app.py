import streamlit as st
import chess
import chess.svg
import base64
from chess_engine import ChessEngine1500
from coach import ChessCoach

# --- PAGE CONFIG ---
st.set_page_config(page_title="Chess AI Coach", layout="wide")

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
                board.push(user_move)
                st.session_state.history.append(f"👤 You: {move_input}")
                
                if not board.is_game_over():
                    # 2. Engine Move
                    with st.spinner("Engine is calculating..."):
                        best_move, analysis = st.session_state.engine.get_best_move(board)
                        
                        # 3. Coach Explanation
                        with st.spinner("Coach is analyzing..."):
                            explanation = st.session_state.coach.explain_move(
                                board.fen(), analysis['move'], analysis['eval']
                            )
                        
                        board.push(best_move)
                        st.session_state.history.append(f"🤖 AI ({analysis['move']}): {explanation}")
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
        st.rerun()

with col_info:
    st.subheader("🎙️ Commentary")
    container = st.container(height=500)
    for msg in reversed(st.session_state.history):
        if "👤" in msg:
            container.markdown(f"**{msg}**")
        else:
            container.info(msg)