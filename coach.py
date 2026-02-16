import os
from dotenv import load_dotenv

# Import specific LangChain requirements
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
        
        # 2. Define the Prompt Template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a chess coach. Explain the move briefly in 1-2 sentences. Don't use markdown bolding."),
            ("user", "Board FEN: {fen}. Move: {move}. Score: {score}. Why is this move good?")
        ])
        
        # 3. Create the Chain
        self.chain = self.prompt | self.llm | self.parser

    def explain_move(self, fen: str, move_san: str, score: int) -> str:
        try:
            response = self.chain.invoke({
                "fen": fen,
                "move": move_san,
                "score": score
            })
            return response
        except Exception as e:
            return f"Coach is offline: {str(e)}"