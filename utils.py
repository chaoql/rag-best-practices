from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core import StorageContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import load_index_from_storage
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.node_parser import SentenceSplitter


def load_hybrid_data(input_file, persist_dir):
    documents = SimpleDirectoryReader(input_files=input_file).load_data()
    # Sliding windows chunking & Extract nodes from documents
    node_parser = SentenceWindowNodeParser.from_defaults(
        # how many sentences on either side to capture
        window_size=3,
        # the metadata key that holds the window of surrounding sentences
        window_metadata_key="window",
        # the metadata key that holds the original sentence
        original_text_metadata_key="original_sentence",
    )
    nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)

    # 创建一个持久化的索引到磁盘
    client = QdrantClient(path=persist_dir)
    # 创建启用混合索引的向量存储
    vector_store = QdrantVectorStore(
        "test", client=client, enable_hybrid=True, batch_size=20
    )
    try:
        storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=persist_dir)
        index = load_index_from_storage(storage_context, show_progress=True)
    except:
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            show_progress=True,
        )
        index.storage_context.persist(persist_dir=persist_dir)
    return index, nodes


def load_txt_data(input_file, persist_dir, with_sliding_window: bool, chunk_size=512, chunk_overlap=128):
    documents = SimpleDirectoryReader(input_files=input_file).load_data()
    if with_sliding_window:
        # Sliding windows chunking & Extract nodes from documents
        node_parser = SentenceWindowNodeParser.from_defaults(
            # how many sentences on either side to capture
            window_size=3,
            # the metadata key that holds the window of surrounding sentences
            window_metadata_key="window",
            # the metadata key that holds the original sentence
            original_text_metadata_key="original_sentence",
        )
    else:
        node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=128)

    nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)

    # indexing & storing
    try:
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context, show_progress=True)
    except:
        index = VectorStoreIndex(nodes=nodes, show_progress=True)
        index.storage_context.persist(persist_dir=persist_dir)
    return index, nodes
