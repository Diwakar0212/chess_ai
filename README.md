# ♟️ Chess AI with LangChain Coach

An interactive chess application powered by a custom 1500 Elo chess engine and an AI coach built with LangChain. Play against the AI, get move explanations, and ask follow-up questions about the game!

## 🌟 Features

- **Chess Engine**: Custom-built 1500 Elo strength engine with alpha-beta pruning, transposition tables, and piece-square tables
- **AI Chess Coach**: Powered by LangChain with support for Ollama (local) or OpenAI models
- **Move Explanations**: Get instant explanations for every AI move
- **Cross-Questioning**: Ask follow-up questions about any position or move
- **Interactive UI**: Clean Streamlit-based interface with real-time board visualization
- **Game Commentary**: Complete move history with AI insights

## 🚀 Installation

### Prerequisites

- Python 3.8+
- (Optional) Ollama installed locally for local LLM support

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Diwakar0212/chess_ai.git
   cd chess_ai
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # or
   source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure AI Provider (Optional)**
   
   Create a `.env` file in the project root:
   
   **For OpenAI:**
   ```env
   OPENAI_API_KEY=your_api_key_here
   ```
   
   **For Ollama (default):**
   - Install [Ollama](https://ollama.ai/)
   - Pull the model: `ollama pull llama3.2`

## 🎮 Usage

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Access the app**
   - Open your browser to `http://localhost:8502`

3. **Play Chess**
   - Enter moves in standard algebraic notation (e.g., `e4`, `Nf3`, `O-O`)
   - Click "Make Move" to play
   - AI will respond with its move and explanation

4. **Ask Questions**
   - After each move, ask questions about the position
   - Example: "What if I played Qh5 instead?"
   - Chat history maintains context for follow-up questions

## 📁 Project Structure

```
chess_ai/
├── app.py              # Streamlit UI and game logic
├── chess_engine.py     # Chess engine implementation
├── coach.py            # LangChain AI coach
├── requirements.txt    # Python dependencies
├── .gitignore         # Git ignore file
└── README.md          # This file
```

## 🛠️ Configuration

### Switching AI Providers

In `app.py`, modify the coach initialization:

```python
# For Ollama (default)
st.session_state.coach = ChessCoach(provider="ollama", model_name="llama3.2")

# For OpenAI
st.session_state.coach = ChessCoach(provider="openai", model_name="gpt-4o")
```

### Engine Difficulty

Adjust the engine depth in `app.py`:

```python
st.session_state.engine = ChessEngine1500(max_depth=3)  # 1-5, higher = stronger
```

## 📦 Dependencies

- `streamlit` - Web interface
- `python-chess` - Chess logic and move validation
- `ollama` - Local LLM support
- `langchain-openai` - OpenAI integration
- `langchain-core` - LangChain core functionality
- `langchain-community` - Community integrations
- `python-dotenv` - Environment variable management

## 🎯 Features in Detail

### Chess Engine
- **Strength**: ~1500 Elo
- **Search Algorithm**: Minimax with alpha-beta pruning
- **Optimizations**: 
  - Transposition tables
  - Killer move heuristic
  - Quiescence search
  - Null move pruning
  - Iterative deepening

### AI Coach
- **Contextual Analysis**: Understands position and move significance
- **Conversation Memory**: Maintains chat history for follow-up questions
- **Multi-Provider**: Works with both local (Ollama) and cloud (OpenAI) models
- **Real-time Explanations**: Instant move analysis

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Chess logic powered by [python-chess](https://python-chess.readthedocs.io/)
- AI capabilities by [LangChain](https://www.langchain.com/)
- UI built with [Streamlit](https://streamlit.io/)

## 📧 Contact

Created by [@Diwakar0212](https://github.com/Diwakar0212)

---

**Enjoy your game! ♟️🤖**
