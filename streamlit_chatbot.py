import os
import sys
import logging
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, AuthenticationError
import tiktoken
import json
from datetime import datetime
import streamlit as st

# Load variables from .env file
load_dotenv() 

# Global Variables
DEFAULT_API_KEY = os.getenv("API_KEY")
DEFAULT_BASE_URL = os.getenv("BASE_URL")
DEFAULT_MODEL = os.getenv("MODEL")
DEFAULT_SYSTEM_MESSAGE = """You are a senior surgeon answering medical students' questions.
                             Give concise, direct, clinically accurate answers. Start with the main answer first. 
                             Use short paragraphs or brief bullet points only when helpful. 
                             Do not give long lectures unless the user asks for more detail."""
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 400
DEFAULT_SEED = 12345
DEFAULT_TOKEN_BUDGET = 4096

# Sensible CLI defaults
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 3

logger = logging.getLogger(__name__)

st.title("Surgical Chatbot")

# Sidebar
st.sidebar.header("Manage chatbot parameters")
max_tokens_per_message = st.sidebar.slider("Max tokens per message", min_value=100, max_value=500, value=400, step=100)
max_tokens_per_conversation = st.sidebar.slider("Max tokens per conversation", min_value=1024, 
        max_value=8192, value=4096, step=1024)
temperature = st.sidebar.slider("Temperature", min_value=0.1, max_value=1.0, value=1.0, step=0.1)
persona = st.sidebar.selectbox("Choose a persona", ["Friendly", "Stern"], index=1)

clear_history = st.sidebar.button("Clear chat history")
if clear_history:
    clear_chat_history()


def configure_logging() -> None:
    """Configure console and file logging for the CLI."""

    handlers: list[logging.Handler] = []

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    handlers.append(console)

    file_handler = logging.FileHandler("chatbot.log", mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    handlers.append(file_handler)

    logging.basicConfig(level=logging.DEBUG, handlers=handlers, force=True)

    # Keep your app verbose, but reduce third-party noise.
    logging.getLogger("openai").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logger.info("======================================================")
    logger.info("               NEW CHATBOT SESSION")
    logger.info("======================================================")


# Conversation Manager class
class ConversationManager:
    def __init__(self, api_key=DEFAULT_API_KEY, base_url=DEFAULT_BASE_URL, model=DEFAULT_MODEL, 
                 temperature=None, max_tokens=None, system_message=None, timeout=DEFAULT_TIMEOUT_SECONDS, 
                 max_retries=DEFAULT_MAX_RETRIES, seed=DEFAULT_SEED, token_budget=None, history_file=None):
        
        # instance attributes
        self.api_key = api_key 
        self.base_url = base_url 
        self.model = model 
        self.temperature = temperature if temperature is not None else DEFAULT_TEMPERATURE
        self.max_tokens = max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS
        self.system_message = system_message if system_message is not None else DEFAULT_SYSTEM_MESSAGE
        self.timeout = timeout 
        self.max_retries = max_retries 
        self.seed = seed 
        
        self.token_budget = token_budget if token_budget is not None else DEFAULT_TOKEN_BUDGET

        self.history_file = history_file or self._generate_history_filename()
        self.conversation_history = self._default_history()
        self._load_conversation_history()
               
        self.client = OpenAI(
            api_key=self.api_key, 
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            )