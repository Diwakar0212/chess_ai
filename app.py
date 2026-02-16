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
if "coach_explanation" not in st.session_state:
    st.session_state.coach_explanation = ""
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

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
                            st.session_state.coach_explanation = explanation
                            # Clear previous chat when new move is analyzed
                            st.session_state.chat_messages = []
                        
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
        st.session_state.coach_explanation = ""
        st.session_state.chat_messages = []
        st.session_state.coach.clear_history()
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
    
    if st.session_state.coach_explanation:
        # Display chat history
        chat_container = st.container(height=200)
        for msg in st.session_state.chat_messages:
            if msg["role"] == "user":
                chat_container.markdown(f"**You:** {msg['content']}")
            else:
                chat_container.info(f"**Coach:** {msg['content']}")
        
        # Question input
        user_question = st.text_input("Ask about the last move:", key="follow_up_question", placeholder="e.g., What if I played differently?")
        
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