import os
import sys
import logging
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, AuthenticationError
import tiktoken
import json
#from datetime import datetime
import streamlit as st

# Load variables from .env file
load_dotenv() 

# Global Variables
DEFAULT_API_KEY = os.getenv("API_KEY")
DEFAULT_BASE_URL = os.getenv("BASE_URL")
DEFAULT_MODEL = os.getenv("MODEL")
DEFAULT_SYSTEM_MESSAGE = (
    "You are a senior surgeon answering medical students' questions. "
    "Give concise, direct, clinically accurate answers. Start with the main answer first. "
    "Use short paragraphs or brief bullet points only when helpful. "
    "Do not give long lectures unless the user asks for more detail."
)
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 400
DEFAULT_SEED = 12345
DEFAULT_TOKEN_BUDGET = 4096
DEFAULT_PERSONA = "Friendly"

# API/client
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 3

logger = logging.getLogger(__name__)


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
    def __init__(self, temperature=None,  max_tokens=None, token_budget=None, persona=None):
        
        # instance attributes
        self.api_key = DEFAULT_API_KEY 
        self.base_url = DEFAULT_BASE_URL 
        self.model = DEFAULT_MODEL 
        self.temperature = temperature if temperature is not None else DEFAULT_TEMPERATURE
        self.max_tokens = max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS
        self.timeout = DEFAULT_TIMEOUT_SECONDS 
        self.max_retries = DEFAULT_MAX_RETRIES 
        self.seed = DEFAULT_SEED 
        
        self.token_budget = token_budget if token_budget is not None else DEFAULT_TOKEN_BUDGET

        self.persona = persona if persona is not None else DEFAULT_PERSONA

        # persona system messages
        self.persona_system_messages = {
            "Friendly": ("You are a senior surgeon answering medical students' questions. "
                        "Give concise, direct, clinically accurate answers. Start with the main answer first. " 
                        "Use short paragraphs or brief bullet points only when helpful. "
                        "Do not give long lectures unless the user asks for more detail. "
                        "You are always trying to be helpful."
            ),
            "Stern": (
                    "You are a no-nonsense professor of surgery. "
                    "You give short, direct answers. "
                    "If the student is incorrect, correct them firmly but professionally."
            ),
            
        }

        if self.persona not in self.persona_system_messages:
            raise ValueError(f"Unknown persona: {self.persona}")

        self.system_message = self.persona_system_messages[self.persona]
        self.conversation_history = self._default_history()
        #self._load_conversation_history()
               
        self.client = OpenAI(
            api_key=self.api_key, 
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            )
        
    def chat_completion(self, prompt: str) -> str:
                
        # Reserve room for the model's reply.
        safety_margin = 100
        input_budget = self.token_budget - self.max_tokens - safety_margin   

        if input_budget <= 0:
            raise ValueError(
                f"Invalid token settings: token_budget = {self.token_budget}, "
                f"max_tokens = {self.max_tokens}, safety_margin = {safety_margin}. "
                f"input budget = {input_budget}. "
                "No room remains for input."
            )

        prompt_tokens = self._count_tokens(prompt)

        logger.debug("------------------------------------------------------")
        logger.debug("New user prompt")

        logger.debug(
            "New prompt received | prompt_tokens=%s max_tokens=%s",
            prompt_tokens,
            self.max_tokens
        )
        
        self.conversation_history.append({
            "role": "user",
            "content": prompt
        })

        logger.debug("Prompt preview: %s", prompt[:120])

        pre_trim_tokens = self._total_tokens_used()
        pre_trim_messages = len(self.conversation_history)

        # Trim so the outbound request stays within budget.
        removed_before_request = self._enforce_input_budget(input_budget) 

        # Save the user prompt
        #self._save_conversation_history()

        post_trim_tokens = self._total_tokens_used()
        post_trim_messages = len(self.conversation_history)

        logger.debug(
            "Pre-request trim | removed_messages=%s tokens_before=%s tokens_after=%s messages_before=%s messages_after=%s input_budget=%s token_budget=%s",
            removed_before_request,
            pre_trim_tokens,
            post_trim_tokens,
            pre_trim_messages,
            post_trim_messages,
            input_budget,
            self.token_budget
        )

        if post_trim_tokens > input_budget:
            logger.warning(
                "History still exceeds input budget after trimming | total=%s input_budget=%s",
                post_trim_tokens,
                input_budget
            )
            raise ValueError(
                f"Latest prompt exceeds input budget after trimming "
                f"(tokens={post_trim_tokens}, limit={input_budget}). "
                "Try reducing prompt length or increasing token budget."
            )

        logger.debug(
            "Sending request | model=%r max_tokens=%s messages=%s",
            self.model,
            self.seed,            
            self. max_tokens,
            len(self.conversation_history)
        )

        try:        
            response = self.client.chat.completions.create(
                model = self.model,
                seed = self.seed,
                messages = self.conversation_history,
                temperature = self.temperature,
                max_tokens = self.max_tokens,
            )

        except Exception:
            logger.exception("Chat completion request failed")
            self._debug_print_history()
            raise

        if response.usage:
            logger.debug(
                "API token usage | prompt=%s completion=%s total=%s",
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                response.usage.total_tokens,
            )

        content = response.choices[0].message.content if response.choices else None

        assistant_tokens = self._count_tokens(content) if content else 0

        logger.debug(
            "Received response | has_choices=%s content_chars=%s assistant_tokens=%s",
            bool(response.choices),
            len(content) if content else 0,
            assistant_tokens
        )

        if content:
            self.conversation_history.append({
                "role": "assistant", 
                "content": content
            })

            pre_response_trim_tokens = self._total_tokens_used()
            pre_response_trim_messages = len(self.conversation_history)

            # Keep stored history under the overall token budget.
            removed_after_response = self._enforce_storage_budget(self.token_budget) 

            post_response_trim_tokens = self._total_tokens_used()
            post_response_trim_messages = len(self.conversation_history)

            logger.debug(
                "Post-response trim | removed_messages=%s tokens_before=%s tokens_after=%s messages_before=%s messages_after=%s token_budget=%s",
                removed_after_response,
                pre_response_trim_tokens,
                post_response_trim_tokens,
                pre_response_trim_messages,
                post_response_trim_messages,
                self.token_budget
            )

            # save the conversation history
            #self._save_conversation_history()
        
        # print the conversation history
        self._debug_print_history()

        return content or "" 
    
    def _add_message(self, sender: str, message: str):
        self.conversation_history.append({"sender": sender, "message": message})
    
    def _count_tokens(self, text: str) -> int:
        """Takes the user prompt and returns the number of tokens used"""
        try:
            encoding = tiktoken.encoding_for_model(self.model)

        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        tokens = encoding.encode(text)
        return len(tokens)
        
    def _total_tokens_used(self) -> int:
        """Approximate total tokens in conversation_history based on message content only."""
        return sum(self._count_tokens(message['content']) for message in self.conversation_history)
    
    def _enforce_input_budget(self, budget_limit: int) -> int:
        """
        Trim oldest middle messages until history fits within budget_limit.

        Preserves:
        - the system message at index 0
        - the most recent user prompt at the end of the history

        Intended for use before sending a request to the API.
        """

        removed_count = 0
        total_tokens_used = self._total_tokens_used()

        while total_tokens_used > budget_limit:
            # Preserve [system, latest_user]
            if len(self.conversation_history) <= 2:
                break

            # Remove oldest complete turn if possible
            if self.conversation_history[1]["role"] == "user":
                if (
                    len(self.conversation_history) >= 3
                    and self.conversation_history[2]["role"] == "assistant"
                ):
                    del self.conversation_history[1:3]
                    removed_count += 2
                else:
                    del self.conversation_history[1]
                    removed_count += 1
            else:
                # Orphan assistant message
                del self.conversation_history[1]
                removed_count += 1

            total_tokens_used = self._total_tokens_used()

        return removed_count
    
    def _enforce_storage_budget(self, budget_limit: int) -> int:
        """
        Trim oldest middle messages until history fits within budget_limit.

        Preserves:
        - the system message at index 0
        - the most recent complete turn at the end of the history
        (latest user + latest assistant)

        Intended for use after receiving and appending the assistant response.
        """

        removed_count = 0
        total_tokens_used = self._total_tokens_used()

        while total_tokens_used > budget_limit:
            # Preserve [system, latest_user, latest_assistant]
            if len(self.conversation_history) <= 3:
                break

            # Remove oldest complete turn if possible
            if self.conversation_history[1]["role"] == "user":
                if (
                    len(self.conversation_history) >= 3
                    and self.conversation_history[2]["role"] == "assistant"
                ):
                    del self.conversation_history[1:3]
                    removed_count += 2
                else:
                    del self.conversation_history[1]
                    removed_count += 1
            else:
                # Orphan assistant message
                del self.conversation_history[1]
                removed_count += 1

            total_tokens_used = self._total_tokens_used()

        return removed_count
        
    def clear_chat_history(self):
        self.chat_history = []

    def _debug_print_history(self):
        """Print the conversation history in a readable format for debugging."""
        logger.debug(
            "---- Conversation History (%s messages) ----",
            len(self.conversation_history)
        )

        total_tokens = 0

        for i, msg in enumerate(self.conversation_history):
            role = msg["role"]
            content = msg["content"]
            tokens = self._count_tokens(content)
            total_tokens += tokens
            logger.debug(
                "%3s | %-9s | %4s tokens | %s", 
                i, 
                role, 
                tokens,
                content[:60] if content else "")
        
        logger.debug("Total messages: %s", len(self.conversation_history))
        logger.debug("Approx total tokens: %s", total_tokens)
        logger.debug("Token usage: %s / %s", total_tokens, self.token_budget)
        logger.debug("------------------------------")

    def _default_history(self) -> dict:
        return [
            {
                "role": "system",
                "content": self.system_message
            }
        ]
    
    def _save_conversation_history(self):
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(self.conversation_history, f, indent=4, ensure_ascii=False)
        

def main():
    configure_logging()

    st.title("Surgical Chatbot")

    # Sidebar
    st.sidebar.header("Manage chatbot parameters")
    st.sidebar.slider(
        "Max tokens per message", 
        min_value=100, 
        max_value=500, 
        value=400, 
        step=100,
        key="max_tokens", #automatically creates and maintains: st.session_state["max_tokens"]
        )
    
    st.sidebar.slider(
        "Max tokens per conversation", 
        min_value=1024, 
        max_value=8192, 
        value=4096, 
        step=1024,
        key="token_budget", #automatically creates and maintains: st.session_state["token_budget"]
        )
    
    st.sidebar.slider(
        "Temperature",
        min_value=0.1, 
        max_value=1.0, 
        value=0.1, 
        step=0.1,
        key="temperature", #automatically creates and maintains: st.session_state["temperature"]
    )
    st.sidebar.selectbox(
        "Choose a persona", 
        ["Friendly", "Stern"], 
        index=0,
        key="persona", #automatically creates and maintains: st.session_state["persona"]
        )

    clear_history = st.sidebar.button("Clear chat history")
       
    try:
        cm = ConversationManager(           
            temperature = st.session_state.temperature,
            max_tokens = st.session_state.max_tokens,            
            token_budget = st.session_state.token_budget,
            persona = st.session_state.persona
        )
    except ValueError as e:
        logger.exception("Configuration error")
        st.error(f"Configuration error: {e}")
        st.stop()

    logger.debug(
        "ConversationManager config: "
        "temperature=%s max_tokens=%s token_budget=%s persona=%s ",       
        cm.temperature,
        cm.max_tokens,
        cm.token_budget,
        cm.persona
    )

    if clear_history:
        cm.clear_chat_history()

    with st.chat_message("assistant"):
        st.write("Ask a question related to surgery.")

   
    try:
        prompt = st.chat_input("Ask a question")
        #prompt = prompt.strip()
        if not prompt:
            st.stop()
            
        if prompt.lower() in {"exit", "quit"}:
            st.stop()
        ai_response = cm.chat_completion(prompt)
        if ai_response:
            for chat in cm.conversation_history[1:]:
                with st.chat_message(chat["role"]):
                    st.write(chat["content"])


    except AuthenticationError:
        logger.exception("Authentication failed")
        st.error("Authentication failed. Check your API key.")
        st.stop()

    except RateLimitError:
        logger.exception("Rate limit exceeded")
        st.error("Rate limit exceeded. Please try again later.")
        st.stop()

    except Exception:
        logger.exception("Unexpected error")
        st.error("An unexpected error occurred.")
        st.stop()
    

if __name__ == "__main__":
    main()