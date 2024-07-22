from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core import PromptTemplate, get_response_synthesizer, StorageContext, VectorStoreIndex, \
    SimpleDirectoryReader, Settings
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core import load_index_from_storage
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import VectorIndexRetriever
import Stemmer
from custom.retriever import CustomRetriever
import warnings

warnings.filterwarnings('ignore')
with_hyde = False  # 是否采用假设文档
persist_dir = "storeQ"  # 向量存储地址
hybrid_search = True  # 是否采用混合检索

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

# load data
documents = SimpleDirectoryReader("./data").load_data()

# Sliding windows chunking & Extract nodes from documents
node_parser = SentenceWindowNodeParser.from_defaults(
    # how many sentences on either side to capture
    window_size=3,
    # the metadata key that holds the window of surrounding sentences
    window_metadata_key="window",
    # the metadata key that holds the original sentence
    original_text_metadata_key="original_sentence",
)
nodes = node_parser.get_nodes_from_documents(documents, show_progress=False)

# indexing & storing
try:
    storage_context = StorageContext.from_defaults(
        docstore=SimpleDocumentStore.from_persist_dir(persist_dir=persist_dir),
        vector_store=SimpleVectorStore.from_persist_dir(persist_dir=persist_dir),
        index_store=SimpleIndexStore.from_persist_dir(persist_dir=persist_dir),
    )
    index = load_index_from_storage(storage_context)
except:
    index = VectorStoreIndex(nodes=nodes)
    index.storage_context.persist(persist_dir=persist_dir)

# prompt
query_str = "How many people are on the deck after ten o'clock?"
# query_str = "what is computer science?"
qa_prompt_tmpl_str = """
Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {query_str}
Answer: 
"""
qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

# Build query_engine
if hybrid_search:
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=2,
        stemmer=Stemmer.Stemmer("english"),
        language="english",
    )
    vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=2)
    custom_retriever = CustomRetriever(vector_retriever, bm25_retriever)
    query_engine = RetrieverQueryEngine.from_args(
        # 自定义prompt Template
        text_qa_template=qa_prompt_tmpl,
        # hybrid search
        retriever=custom_retriever,
        # the target key defaults to `window` to match the node_parser's default
        node_postprocessors=[
            # LLM reranker
            LLMRerank(top_n=2, llm=Settings.llm),
            # replace the sentence in each node with its surrounding context.
            MetadataReplacementPostProcessor(target_metadata_key="window"),
        ],
        # 对上下文进行简单摘要，当上下文较长或检索到的块较多时应使用tree_summary，否则使用simple_summary。
        response_synthesizer=get_response_synthesizer(
            response_mode=ResponseMode.TREE_SUMMARIZE),
    )
else:
    # Build a tree index over the set of candidate nodes, with a summary prompt seeded with the query. with LLM reranker
    query_engine = index.as_query_engine(similarity_top_k=2,
                                         text_qa_template=qa_prompt_tmpl,
                                         node_postprocessors=[
                                             LLMRerank(top_n=2, llm=Settings.llm),
                                             MetadataReplacementPostProcessor(target_metadata_key="window"),
                                         ],
                                         response_synthesizer=get_response_synthesizer(
                                             response_mode=ResponseMode.TREE_SUMMARIZE),
                                         )

# HyDE(当问题较为简单时，不需要该模块参与)
if with_hyde:
    hyde = HyDEQueryTransform(include_original=True)
    query_engine = TransformQueryEngine(query_engine, hyde)

# response
response = query_engine.query(query_str)
print(f"Question: {str(query_str)}")
print("------------------")
print(f"Response: {str(response)}")
print("------------------")
try:
    window = response.source_nodes[0].node.metadata["window"]  # 长度为3的窗口，包含了文本两侧的上下文。
    sentence = response.source_nodes[0].node.metadata["original_sentence"]  # 检索到的文本
    print(f"Window: {window}")
    print("------------------")
    print(f"Original Sentence: {sentence}")
    print("------------------")
except:
    pass

"""
Question: How many people are on the deck after ten o'clock?
------------------
Response: After ten o'clock, there were only three or five pairs of men and women on the deck.
------------------
Window: Sun with two empty chairs.  Fortunately, the cigarette incident just now fell into their eyes.  That evening, there was a sea breeze and the boat was a bit bumpy.  After ten o'clock, there were only three or five pairs of men and women on the deck, all hiding in the dark shadows that could not be illuminated by the lights, whispering sweet words.  Fang Hongjian and Miss Bao walked side by side without saying a word.  A big wave shook the hull of the ship, and Miss Bao couldn't stand steadily.  Fang Hongjian hooked her waist and leaned against the railing, kissing her greedily. 
------------------
Original Sentence: After ten o'clock, there were only three or five pairs of men and women on the deck, all hiding in the dark shadows that could not be illuminated by the lights, whispering sweet words. 
------------------
"""
