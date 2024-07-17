from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext
import warnings
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.prompts import LangchainPromptTemplate
from langchain import hub
from llama_index.core import PromptTemplate
warnings.filterwarnings('ignore')

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

# indexing
try:
    storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir="storeQ")
    index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)
except:
    index = VectorStoreIndex(nodes=nodes)
    index.storage_context.persist(persist_dir="storeQ")

# prompt
query_str = "How old is the boy's mother?"
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

# replace the sentence in each node with its surrounding context.
query_engine = index.as_query_engine(similarity_top_k=2,
                                     # the target key defaults to `window` to match the node_parser's default
                                     node_postprocessors=[
                                         MetadataReplacementPostProcessor(target_metadata_key="window")
                                     ],
                                     # 自定义prompt Template
                                     text_qa_template=qa_prompt_tmpl,
                                     )

# 响应
response = query_engine.query(query_str)
window = response.source_nodes[0].node.metadata["window"]  # 长度为3的窗口，包含了文本两侧的上下文。
sentence = response.source_nodes[0].node.metadata["original_sentence"]  # 检索到的文本

print(f"Question: {str(query_str)}")
print("------------------")
print(f"Response: {str(response)}")
print("------------------")
print(f"Window: {window}")
print("------------------")
print(f"Original Sentence: {sentence}")

"""
Question: How old is the boy's mother?
------------------
Response: The boy's mother is described as having "thirty outside" which likely refers to her age. However, without further context or clarification from the text, it is impossible to determine exactly how old she is. The sentence suggests that she has been working hard and may be tired, but this does not necessarily imply a specific age.
------------------
Window: She had removed her dark glasses, and her eyebrows were clear, but her lips were too thin, and the lipstick was not rich enough.  If she stood up from the canvas recliner, she would look thin, perhaps the lines of her silhouette were too hard, like the strokes of a square-tipped fountain pen.  She looked twenty-five or twenty-six years old, but the age of a new school woman is like the age of an old-fashioned woman's wedding invitation, which requires what the expert scientist calls extrinsic evidence to determine its authenticity, and which cannot be seen by itself.  The boy's mother has thirty outside, wearing a half old black cheongsam, full of labor and tiredness, coupled with the natural upside down eyebrows, the more sad and pathetic.  The child is less than two years old, collapsed nose, two slits in the eyes, eyebrows high above, and eyes far away from each other to suffer from lovesickness, like the Chinese face in the newspaper caricature.  He had just begun to walk, and was constantly running about; his mother held a leash on him, and pulled him back when he could not run more than three or four steps.  His mother, who was afraid of the heat, was tired of pulling him, but she was also concerned about her husband's success down there and couldn't stop scolding the boy for being a nuisance. 
------------------
Original Sentence: The boy's mother has thirty outside, wearing a half old black cheongsam, full of labor and tiredness, coupled with the natural upside down eyebrows, the more sad and pathetic.
"""