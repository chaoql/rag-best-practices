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
from custom.glmfz import ChatGLM
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from dotenv import load_dotenv, find_dotenv
from dotenv import dotenv_values

warnings.filterwarnings('ignore')

_ = load_dotenv(find_dotenv())  # 导入环境
config = dotenv_values(".env")
# 设置参数
with_hyde = False  # 是否采用假设文档
persist_dir = "storeQ"  # 向量存储地址
hybrid_search = True  # 是否采用混合检索
top_k = 3
response_mode = ResponseMode.TREE_SUMMARIZE  # 最佳实践为为TREE_SUMMARIZE
with_query_classification = True  # 是否对输入的问题进行分类
# 根据如下description选择是否需要进行检索增强生成
rag_description = "Useful for answering questions that require specific contextual knowledge to be answered accurately."
norag_rag_description = ("Used to answer questions that do not require specific contextual knowledge to be answered "
                         "accurately.")
query_str = "How many people are on the deck after ten o'clock?"
# query_str = "Who did Fang Hongjian kiss?"
# query_str = "what is computer?"

# 加载嵌入模型
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-large-zh-v1.5",
    cache_folder="./BAAI/",
    embed_batch_size=128,
    local_files_only=True,  # 仅加载本地模型，不尝试下载
    device="cuda",
)

# 加载大模型
# Settings.llm = Ollama(model="qwen2:1.5b", request_timeout=30.0, temperature=0)
Settings.llm = ChatGLM(
    api_key=config["GLM_KEY"],
    model="glm-4",
    api_base="https://open.bigmodel.cn/api/paas/v4/",
    is_chat_model=True,
)
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
qa_prompt_tmpl_str = """
Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {query_str}
Answer: 
"""
simple_qa_prompt_tmpl_str = """
Please answer the query.
Query: {query_str}
Answer: 
"""
qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
simple_qa_prompt_tmpl = PromptTemplate(simple_qa_prompt_tmpl_str)  # norag

# Build query_engine
if hybrid_search:
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=top_k,
        stemmer=Stemmer.Stemmer("english"),
        language="english",
    )
    vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
    custom_retriever = CustomRetriever(vector_retriever, bm25_retriever, mode="AND", alpha=0.3)
    rag_query_engine = RetrieverQueryEngine.from_args(
        # 自定义prompt Template
        text_qa_template=qa_prompt_tmpl,
        # hybrid search
        retriever=custom_retriever,
        # the target key defaults to `window` to match the node_parser's default
        node_postprocessors=[
            # LLM reranker（注意：使用大模型进行重排序时不保证输出可解析）
            # LLMRerank(top_n=top_k, llm=Settings.llm),
            # replace the sentence in each node with its surrounding context.
            MetadataReplacementPostProcessor(target_metadata_key="window"),
        ],
        # 对上下文进行简单摘要，当上下文较长或检索到的块较多时应使用tree_summary，否则使用simple_summary。
        response_synthesizer=get_response_synthesizer(
            response_mode=response_mode),
    )
else:
    # Build a tree index over the set of candidate nodes, with a summary prompt seeded with the query. with LLM reranker
    rag_query_engine = index.as_query_engine(similarity_top_k=top_k,
                                             text_qa_template=qa_prompt_tmpl,
                                             node_postprocessors=[
                                                 # LLMRerank(top_n=top_k, llm=Settings.llm),
                                                 MetadataReplacementPostProcessor(target_metadata_key="window"),
                                             ],
                                             response_synthesizer=get_response_synthesizer(
                                                 response_mode=response_mode),
                                             )
simple_query_engine = index.as_query_engine(similarity_top_k=top_k,
                                            text_qa_template=simple_qa_prompt_tmpl,
                                            response_synthesizer=get_response_synthesizer(
                                                response_mode=ResponseMode.GENERATION),
                                            )

# HyDE(当问题较为简单时，不需要该模块参与)
if with_hyde:
    hyde = HyDEQueryTransform(include_original=True)
    rag_query_engine = TransformQueryEngine(rag_query_engine, hyde)

# Router Query Engine(Query Classification)
rag_tool = QueryEngineTool.from_defaults(
    query_engine=rag_query_engine,
    description=rag_description,
)
simple_tool = QueryEngineTool.from_defaults(
    query_engine=simple_query_engine,
    description=norag_rag_description,
)
query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[
        rag_tool,
        simple_tool,
    ],
)

# response
if with_query_classification:
    response = query_engine.query(query_str)
else:
    response = rag_query_engine.query(query_str)

print(f"Question: {str(query_str)}")
print("------------------")
print(f"Response: {str(response)}")
print("------------------")
if response.metadata['selector_result'].ind == 0:
    window = response.source_nodes[0].node.metadata["window"]  # 长度为3的窗口，包含了文本两侧的上下文。
    sentence = response.source_nodes[0].node.metadata["original_sentence"]  # 检索到的文本
    print(f"Window: {window}")
    print("------------------")
    print(f"Original Sentence: {sentence}")
    print("------------------")

"""
示例1：
Question: How many people are on the deck after ten o'clock?
------------------
Response: After ten o'clock, there were only three or five pairs of men and women on the deck.
------------------
Window: Sun with two empty chairs.  Fortunately, the cigarette incident just now fell into their eyes.  That evening, there was a sea breeze and the boat was a bit bumpy.  After ten o'clock, there were only three or five pairs of men and women on the deck, all hiding in the dark shadows that could not be illuminated by the lights, whispering sweet words.  Fang Hongjian and Miss Bao walked side by side without saying a word.  A big wave shook the hull of the ship, and Miss Bao couldn't stand steadily.  Fang Hongjian hooked her waist and leaned against the railing, kissing her greedily. 
------------------
Original Sentence: After ten o'clock, there were only three or five pairs of men and women on the deck, all hiding in the dark shadows that could not be illuminated by the lights, whispering sweet words. 
------------------
示例2：
Question: Who did Fang Hongjian kiss?
------------------
Response: Fang Hongjian kissed Miss Bao.
------------------
Window: After ten o'clock, there were only three or five pairs of men and women on the deck, all hiding in the dark shadows that could not be illuminated by the lights, whispering sweet words.  Fang Hongjian and Miss Bao walked side by side without saying a word.  A big wave shook the hull of the ship, and Miss Bao couldn't stand steadily.  Fang Hongjian hooked her waist and leaned against the railing, kissing her greedily.  Miss Bao's lips hinted, her body obediently, and this hurried and rough kiss gradually stabilized, growing perfectly close.  Miss Bao deftly pushed away Fang Hongjian's arm, took a deep breath, and said, "I'm suffocating you!  I'm catching a cold and can't breathe in my nose - it's too cheap for you, you haven't begged me to love you yet.
------------------
Original Sentence: Fang Hongjian hooked her waist and leaned against the railing, kissing her greedily. 
------------------
示例3：
Question: what is computer?
------------------
Response: A computer is an electronic device that can be programmed to carry out a sequence of arithmetic or logical operations automatically. It is a versatile and complex tool that has become an integral part of modern life. Computers can perform a wide variety of tasks, such as processing and storing large amounts of information, displaying graphics and text, connecting to the internet, and running applications that help with productivity, creativity, and communication.
At its most basic level, a computer consists of a central processing unit (CPU) that performs most of the calculations, and memory that stores both data and instructions for the CPU. It also includes input devices (like keyboards and mice), output devices (like monitors and printers), and storage devices (like hard drives or solid-state drives) for long-term data storage.
Computers can range in size from large mainframes that fill entire rooms to small and powerful devices like smartphones that can fit in your pocket. They can be general-purpose, like personal computers, or designed for specific tasks, like embedded systems in cars or appliances.
The field of computer science is dedicated to the study of computers, including their design, hardware and software components, applications, and the theoretical limits of computation.
------------------
"""
