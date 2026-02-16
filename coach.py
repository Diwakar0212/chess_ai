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
            ("system", "You are a chess coach. Explain the move briefly in 1-2 sentences. Don't use markdown bolding."),
            ("user", "Board FEN: {fen}. Move: {move}. Score: {score}. Why is this move good?")
        ])
        
        # 3. Create the Chain for move explanation
        self.chain = self.prompt | self.llm | self.parser
        
        # 4. Chat history for cross-questioning
        self.chat_history = []
        self.current_fen = None

    def explain_move(self, fen: str, move_san: str, score: int) -> str:
        try:
            # Reset chat history when analyzing a new position
            self.current_fen = fen
            self.chat_history = []
            
            response = self.chain.invoke({
                "fen": fen,
                "move": move_san,
                "score": score
            })
            
            # Store the initial explanation in chat history
            self.chat_history.append(HumanMessage(content=f"Board FEN: {fen}. Move: {move_san}. Score: {score}. Why is this move good?"))
            self.chat_history.append(AIMessage(content=response))
            
            return response
        except Exception as e:
            return f"Coach is offline: {str(e)}"
    
    def ask_followup(self, question: str) -> str:
        """
        Ask a follow-up question about the current position.
        """
        try:
            if not self.current_fen:
                return "Please analyze a move first before asking follow-up questions."
            
            # Create a prompt with chat history for context
            messages = [
                ("system", f"You are a chess coach analyzing position: {self.current_fen}. Answer the user's question briefly and clearly. Don't use markdown bolding.")
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
        """Clear the chat history."""
        self.chat_history = []
        self.current_fen = None