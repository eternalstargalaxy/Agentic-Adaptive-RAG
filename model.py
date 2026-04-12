import os

from dotenv import load_dotenv

from graph.embeddings import BGEM3BiEncoderEmbeddings

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None

# Load environment variables
load_dotenv()

class MissingGoogleModel:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def _raise(self):
        raise RuntimeError(
            f"{self.model_name} requires GOOGLE_API_KEY to be set in the environment."
        )

    def with_structured_output(self, *args, **kwargs):
        self._raise()

    def invoke(self, *args, **kwargs):
        self._raise()

    def __or__(self, other):
        self._raise()


google_api_key = os.getenv("GOOGLE_API_KEY")

if google_api_key and ChatGoogleGenerativeAI is not None:
    llm_model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=google_api_key,
        temperature=0.2,
    )

    grader_model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=google_api_key,
        temperature=0,
    )
else:
    llm_model = MissingGoogleModel("llm_model")
    grader_model = MissingGoogleModel("grader_model")

embed_model = BGEM3BiEncoderEmbeddings(
    model_name=os.getenv("DENSE_MODEL_NAME", "BAAI/bge-m3"),
    query_adapter_path=os.getenv("QUERY_TOWER_ADAPTER_PATH"),
    query_instruction=os.getenv(
        "QUERY_TOWER_INSTRUCTION",
        "Represent this medical query for retrieving supporting evidence: ",
    ),
)
