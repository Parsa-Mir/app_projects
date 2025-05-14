from pyapp.model_connection.model import  get_model_embeddings_from_config_dir
from pyapp.config import find_config, get_data_dir
from pyapp.vectordb import get_vector_store
from pathlib import Path
from pyapp.observation.phoneix import traced_agent
from loguru import logger

here = Path(__file__).resolve()
config_dir = find_config(here)
model = get_model_embeddings_from_config_dir(config_dir=config_dir)
data_dir = get_data_dir(here)

vector_store_path = data_dir / "vectordb/test.db"
raw_data_path = data_dir / "raw_data/test.csv"
table_name = "test"

logger.info(f"vector_store_path: {vector_store_path}")
logger.info(f"raw_data_path: {raw_data_path}")
logger.info(f"table_name: {table_name}")

vector_store, conn = get_vector_store(database_path= str(vector_store_path) , table_name=table_name , embedding_model=model)
retriever = vector_store.as_retriever(search_kwargs={"k": 1})

@traced_agent(name="airplane-simple-retriever")
def agent(messages: list[dict], session_id: str) -> str:
    question = messages[-1]["content"]
    return retriever.invoke(question)

def inference():
    import uuid
    session_id = str(uuid.uuid4())

    messages = [
    {"role": "user", "content": "what is the capital of france?"}
    ]
    answer, link = agent(messages, session_id)

    return answer, link




def generate():
    try:
        from langchain_community.document_loaders.csv_loader import CSVLoader
        loader = CSVLoader(file_path=str(raw_data_path.resolve()))
        data = loader.load()
        vector_store.add_documents(data)
    finally:
        conn.close()
