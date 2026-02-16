import os
from dotenv import load_dotenv

# Import specific LangChain requirements
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables from .env
try:
    load_dotenv()
except:
    pass  # .env file is optional

class ChessCoach:
    def __init__(self, provider="ollama", model_name="llama3.2"):
        """
        Initialize the Coach.
        provider: 'ollama' or 'openai'
        model_name: 'llama3.2' for Ollama, 'gpt-4o' for OpenAI
        """
        self.provider = provider
        
        # 1. Select the LLM based on provider
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in .env file")
            self.llm = ChatOpenAI(model=model_name, temperature=0.3)
        else:
            # Default to Ollama (Community package)
            self.llm = ChatOllama(model=model_name, temperature=0.3)

        self.parser = StrOutputParser()
        
        # 2. Define the Prompt Template for move explanation
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a chess coach. Explain the chess move briefly in 1-2 sentences focusing on its strategic purpose. Don't use markdown bolding."),
            ("user", "{player} played: {move}. Evaluation score: {score}. Why is this move good?")
        ])
        
        # 3. Create the Chain for move explanation
        self.chain = self.prompt | self.llm | self.parser
        
        # 4. Chat history for cross-questioning
        self.chat_history = []
        self.current_fen = None
        self.current_move = None
        self.current_score = None
        self.current_player = None
        self.move_history = []  # Track all moves with player info

    def explain_move(self, fen: str, move_san: str, score: int, player: str = "AI") -> str:
        try:
            # Store context for follow-up questions
            self.current_fen = fen
            self.current_move = move_san
            self.current_score = score
            self.current_player = player
            self.chat_history = []
            
            # Add to move history
            self.move_history.append({
                "player": player,
                "move": move_san,
                "score": score
            })
            
            # Keep only last 10 moves
            if len(self.move_history) > 10:
                self.move_history = self.move_history[-10:]
            
            response = self.chain.invoke({
                "player": player,
                "move": move_san,
                "score": score
            })
            
            # Store the initial explanation in chat history
            self.chat_history.append(HumanMessage(content=f"{player} played: {move_san}. Evaluation score: {score}. Why is this move good?"))
            self.chat_history.append(AIMessage(content=response))
            
            return response
        except Exception as e:
            return f"Coach is offline: {str(e)}"
    
    def ask_followup(self, question: str) -> str:
        """
        Ask a follow-up question about the current position or move history.
        """
        try:
            if not self.current_move:
                return "Please analyze a move first before asking follow-up questions."
            
            # Build move history context
            move_context = "\n".join([
                f"{m['player']}: {m['move']} (score: {m['score']})" 
                for m in self.move_history[-5:]  # Last 5 moves
            ])
            
            # Create a prompt with chat history for context
            messages = [
                ("system", f"""You are a chess coach. 

Recent moves:
{move_context}

Last move: {self.current_player} played {self.current_move} (score: {self.current_score})

Answer the user's chess question clearly and briefly. When they ask about "my move" or "their move", refer to the player who made that specific move. Focus on practical chess advice. Don't use markdown bolding.""")
            ]
            
            # Add chat history
            for msg in self.chat_history:
                if isinstance(msg, HumanMessage):
                    messages.append(("user", msg.content))
                else:
                    messages.append(("assistant", msg.content))
            
            # Add new question
            messages.append(("user", question))
            
            # Create temporary chain with history
            temp_prompt = ChatPromptTemplate.from_messages(messages)
            temp_chain = temp_prompt | self.llm | self.parser
            
            response = temp_chain.invoke({})
            
            # Update chat history
            self.chat_history.append(HumanMessage(content=question))
            self.chat_history.append(AIMessage(content=response))
            
            return response
        except Exception as e:
            return f"Coach is offline: {str(e)}"
    
    def clear_history(self):
        """Clear the chat history but keep move history."""
        self.chat_history = []
        # Don't clear move_history so we can still reference previous moves