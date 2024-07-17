from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.legacy.embeddings import FastEmbedEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core import VectorStoreIndex
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import StorageContext
from llama_index.core import PromptTemplate
from tqdm.asyncio import tqdm
import nest_asyncio
from llama_index.legacy.retrievers.bm25_retriever import BM25Retriever
import asyncio
import os
import warnings
from llama_index.core.node_parser import SentenceWindowNodeParser

# node_parser = SentenceWindowNodeParser.from_defaults(
#     # how many sentences on either side to capture
#     window_size=3,
#     # the metadata key that holds the window of surrounding sentences
#     window_metadata_key="window",
#     # the metadata key that holds the original sentence
#     original_text_metadata_key="original_sentence",
# )
# warnings.filterwarnings('ignore')


# 加载嵌入模型
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-large-zh-v1.5",
    cache_folder="./BAAI/",
    embed_batch_size=128,
    local_files_only=True,  # 仅加载本地模型，不尝试下载
    # device="cuda",
)

# 加载大模型
Settings.llm = Ollama(model="qwen2:1.5b", request_timeout=30.0, temperature=0)

# 部署向量数据库
client = qdrant_client.QdrantClient()
vector_store = QdrantVectorStore(client=client, collection_name="paul_graham")

# load data
documents = SimpleDirectoryReader("./data").load_data()

# Transformations
pipeline = IngestionPipeline(transformations=[TokenTextSplitter(chunk_size=512), node_parser])

# Extract nodes from documents
nodes = pipeline.run(documents=documents)

# indexing
try:
    storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir="storeQ_test")
    index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)
except:
    index = VectorStoreIndex(nodes=nodes)
    index.storage_context.persist(persist_dir="storeQ_test")

# prompt
query_str = "什么是裸金属？"

# 响应
# response = Settings.llm.complete(query_str)
# print(response)

query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query(query_str)

print(str(response))
