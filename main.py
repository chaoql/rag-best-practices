from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core import PromptTemplate, get_response_synthesizer, StorageContext, VectorStoreIndex, \
    SimpleDirectoryReader, Settings
from FTEmbed import finetuning_data_preparation, finetuning_embedding, eval_finetuning_embedding
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core.response_synthesizers.type import ResponseMode
from custom.glmfz import ChatGLM
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from dotenv import load_dotenv, find_dotenv
from dotenv import dotenv_values
from custom.query import build_query_engine
from custom.prompt import qa_prompt_tmpl_str, simple_qa_prompt_tmpl_str, rag_description, norag_rag_description
from utils import load_hybrid_data, load_txt_data
import warnings

warnings.filterwarnings('ignore')
# 导入环境
_ = load_dotenv(find_dotenv())
config = dotenv_values(".env")

# 设置参数
with_hyde = False  # 是否采用假设文档
persist_dir = "store"  # 向量存储地址
with_hybrid_search = True  # 是否采用混合检索
# 5选2
top_k = 5
top_k_rerank = 2
with_sliding_window = True  # 是否采用滑动窗口
response_mode = ResponseMode.SIMPLE_SUMMARIZE  # RAG架构，最佳实践为为TREE_SUMMARIZE
with_query_classification = False  # 是否对输入的问题进行分类
with_rerank = True  # 是否采用重排序
with_local_llm = False  # 是否采用本地基于Ollama的大模型
with_Finetuning_embedding = False  # 是否微调嵌入模型
with_Finetuning_embedding_eval = False  # 是否测评微调嵌入模型的命中率
# 提问
# query_str = "Did Fang Hung-chien kiss Miss Bao?"
query_str = "In the text, which lady did Fang Hongjian kiss on the ship, and under what circumstances did it happen?"

# 加载嵌入模型
if with_Finetuning_embedding:
    # 微调需要开启VPN
    finetuning_data_preparation(all_data=["data/testdata.txt"], llm=Settings.llm, verbose=False,
                                train_dataset_dir="ft_data/train_dataset.json",
                                val_dataset_dir="ft_data/val_dataset.json",
                                qa_finetune_train_dataset_dir="ft_data/qa_finetune_train_dataset.json",
                                qa_finetune_val_dataset_dir="ft_data/qa_finetune_val_dataset.json")
    Settings.embed_model = finetuning_embedding(train_dataset_dir="ft_data/train_dataset.json",
                                                val_dataset_dir="ft_data/val_dataset.json",
                                                model_name="BAAI/bge-large-en-v1.5",
                                                model_output_path="BAAI/ft-bge-large-en-v1.5/")
    # 测评微调后的嵌入模型命中率
    if with_Finetuning_embedding_eval:
        eval_finetuning_embedding(embed_model=Settings.embed_model, val_dataset_dir="ft_data/val_dataset.json",
                                  model_name="ft-bge-large-en-v1.5")
else:
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-large-en-v1.5",
        cache_folder="./BAAI/",
        embed_batch_size=128,
        local_files_only=True,  # 仅加载本地模型，不尝试下载
        device="cuda",
    )

# 加载大模型
if with_local_llm:
    Settings.llm = Ollama(model="qwen2:1.5b", request_timeout=30.0, temperature=0)
else:
    Settings.llm = ChatGLM(
        api_key=config["GLM_KEY"],
        model="glm-4",
        api_base="https://open.bigmodel.cn/api/paas/v4/",
        is_chat_model=True,
    )

# load data(是否采用混合检索)
if with_hybrid_search:
    index, nodes = load_hybrid_data(input_file=["data/testdata.txt"], persist_dir="hybrid_Store")
else:
    index, nodes = load_txt_data(input_file=["data/testdata.txt"], persist_dir="hybrid_Store",
                                 with_sliding_window=with_sliding_window, chunk_size=512, chunk_overlap=128)

# prompt
qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
simple_qa_prompt_tmpl = PromptTemplate(simple_qa_prompt_tmpl_str)  # norag

# Build query_engine
rag_query_engine = build_query_engine(index, response_mode, qa_prompt_tmpl, with_hybrid_search, top_k,
                                      top_k_rerank, with_rerank, nodes)
simple_query_engine = index.as_query_engine(similarity_top_k=top_k,
                                            text_qa_template=simple_qa_prompt_tmpl,
                                            response_synthesizer=get_response_synthesizer(
                                                response_mode=ResponseMode.GENERATION),
                                            )

# HyDE(当问题较为简单时，不需要该模块参与)
if with_hyde:
    hyde = HyDEQueryTransform(include_original=True)
    rag_query_engine = TransformQueryEngine(rag_query_engine, hyde)

# response
# Router Query Engine(Query Classification)
if with_query_classification:
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
    response = query_engine.query(query_str)
else:
    response = rag_query_engine.query(query_str)

print(f"Question: {str(query_str)}")
print("------------------")
print(f"Response: {str(response)}")
print("------------------")
if not with_query_classification or response.metadata['selector_result'].ind == 0:
    window = response.source_nodes[0].node.metadata["window"]  # 长度为3的窗口，包含了文本两侧的上下文。
    sentence = response.source_nodes[0].node.metadata["original_sentence"]  # 检索到的文本
    print(f"Window: {window}")
    print("------------------")
    print(f"Original Sentence: {sentence}")
    print("------------------")

"""
示例1：
Question: Did Fang Hung-chien kiss Miss Bao?
------------------
Response: Yes, Fang Hung-chien kissed Miss Bao.
------------------
Window: A big wave shook the hull badly, and Miss Bao could not stand steadily.  Fang hung-chien hooked her waist and stayed by the railing, kissing her greedily.  Miss Bao's lips suggested that the body followed, and this hasty and rude kiss gradually stabilized and grew properly and densely.  Miss Bao deftly pushed off Fang Hung-chien's arm, took a deep breath in her mouth and said, "I'm suffocated by you!  I have a cold and I can't breathe in my nose-it's too cheap for you, and you haven't asked me to love you! "
 "I beg you now, ok?"  It seems that all men who have never been in love, Fang Hung-chien regards the word "love" too noble and serious and refuses to apply it to women casually; He only felt that he wanted Miss Bao and didn't love her, so he was so evasive.
------------------
Original Sentence: Miss Bao deftly pushed off Fang Hung-chien's arm, took a deep breath in her mouth and said, "I'm suffocated by you! 
------------------

示例2：
Question: In the text, which lady did Fang Hongjian kiss on the ship, and under what circumstances did it happen?
------------------
Response: In the text, Fang Hongjian kissed Miss Bao on the ship. It happened in the dark shadows of the deck, after a big wave shook the hull, causing Miss Bao to lose her balance. Fang Hongjian hooked her waist to steady her and then kissed her.
------------------
Window: After ten o'clock, there were only three or five pairs of men and women on the deck, all hiding in the dark shadows where the lights could not shine.  Fang Hung-chien and Miss Bao walked side by side without talking.  A big wave shook the hull badly, and Miss Bao could not stand steadily.  Fang hung-chien hooked her waist and stayed by the railing, kissing her greedily.  Miss Bao's lips suggested that the body followed, and this hasty and rude kiss gradually stabilized and grew properly and densely.  Miss Bao deftly pushed off Fang Hung-chien's arm, took a deep breath in her mouth and said, "I'm suffocated by you!  I have a cold and I can't breathe in my nose-it's too cheap for you, and you haven't asked me to love you! "
------------------
Original Sentence: Fang hung-chien hooked her waist and stayed by the railing, kissing her greedily. 
------------------
"""
