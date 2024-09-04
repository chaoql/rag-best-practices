import json
import os

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode
from llama_index.finetuning import generate_qa_embedding_pairs
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from dotenv import dotenv_values
from custom.glmfz import ChatGLM
from llama_index.core import Settings
from llama_index.finetuning import SentenceTransformersFinetuneEngine
from llama_index.core import ServiceContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from tqdm.notebook import tqdm
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pandas as pd
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def load_corpus(files, verbose=False):
    if verbose:
        print(f"Loading files {files}")

    reader = SimpleDirectoryReader(input_files=files)
    docs = reader.load_data()
    if verbose:
        print(f"Loaded {len(docs)} docs")

    parser = SentenceSplitter()
    nodes = parser.get_nodes_from_documents(docs, show_progress=verbose)

    if verbose:
        print(f"Parsed {len(nodes)} nodes")

    return nodes


def evaluate(
        dataset,
        embed_model,
        top_k=5,
        verbose=False,
):
    corpus = dataset.corpus
    queries = dataset.queries
    relevant_docs = dataset.relevant_docs

    nodes = [TextNode(id_=id_, text=text) for id_, text in corpus.items()]
    index = VectorStoreIndex(
        nodes, embed_model=embed_model, show_progress=True
    )
    retriever = index.as_retriever(similarity_top_k=top_k)

    eval_results = []
    for query_id, query in tqdm(queries.items()):
        retrieved_nodes = retriever.retrieve(query)
        retrieved_ids = [node.node.node_id for node in retrieved_nodes]
        expected_id = relevant_docs[query_id][0]
        is_hit = expected_id in retrieved_ids  # assume 1 relevant doc

        eval_result = {
            "is_hit": is_hit,
            "retrieved": retrieved_ids,
            "expected": expected_id,
            "query": query_id,
        }
        eval_results.append(eval_result)
    return eval_results


def evaluate_st(
        dataset,
        model_id,
        name,
):
    corpus = dataset.corpus
    queries = dataset.queries
    relevant_docs = dataset.relevant_docs

    evaluator = InformationRetrievalEvaluator(
        queries, corpus, relevant_docs, name=name
    )
    model = SentenceTransformer(model_id)
    output_path = "results/"
    Path(output_path).mkdir(exist_ok=True, parents=True)
    return evaluator(model, output_path=output_path)


_ = load_dotenv(find_dotenv())  # 导入环境
config = dotenv_values(".env")
Settings.llm = ChatGLM(
    api_key=config["GLM_KEY"],
    model="glm-4",
    api_base="https://open.bigmodel.cn/api/paas/v4/",
    is_chat_model=True,
)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-large-en-v1.5",
    cache_folder="./BAAI/",
    embed_batch_size=128,
    local_files_only=True,  # 仅加载本地模型，不尝试下载
    device="cuda",
)


def finetuning_data_preparation(all_data: list, llm, verbose: bool = False,
                                train_dataset_dir: str = "ft_data/train_dataset.json",
                                val_dataset_dir: str = "ft_data/val_dataset.json",
                                qa_finetune_train_dataset_dir: str = "ft_data/qa_finetune_train_dataset.json",
                                qa_finetune_val_dataset_dir: str = "ft_data/qa_finetune_val_dataset.json"):
    docs_nodes = load_corpus(all_data, verbose=verbose)
    train_nodes = docs_nodes[:int(2 * len(docs_nodes) / 3)]
    val_nodes = docs_nodes[int(2 * len(docs_nodes) / 3 + 1):]

    # 自带保存
    train_dataset = generate_qa_embedding_pairs(llm=llm, nodes=train_nodes, verbose=verbose,
                                                output_path="ft_data/qa_finetune_train_dataset.json")
    val_dataset = generate_qa_embedding_pairs(llm=llm, nodes=val_nodes, verbose=verbose,
                                              output_path="ft_data/qa_finetune_val_dataset.json")
    train_dataset.save_json(train_dataset_dir)
    val_dataset.save_json(val_dataset_dir)


def finetuning_embedding(train_dataset_dir: str = "ft_data/train_dataset.json",
                         val_dataset_dir: str = "ft_data/val_dataset.json",
                         model_name: str = "BAAI/bge-large-en-v1.5",
                         model_output_path: str = "BAAI/ft-bge-large-en-v1.5"):
    # [Optional] Load
    train_dataset = EmbeddingQAFinetuneDataset.from_json(train_dataset_dir)
    val_dataset = EmbeddingQAFinetuneDataset.from_json(val_dataset_dir)
    finetune_engine = SentenceTransformersFinetuneEngine(
        train_dataset,
        model_id=model_name,
        model_output_path=model_output_path,
        val_dataset=val_dataset,
    )
    if not os.path.exists(model_output_path):  # 如果已经存在微调好的模型就不用重新微调了
        finetune_engine.finetune()
    embed_model = finetune_engine.get_finetuned_model()
    print(embed_model)
    return embed_model


def eval_finetuning_embedding(embed_model, val_dataset_dir: str = "ft_data/val_dataset.json",
                              model_name: str = "bge-large-en-v1.5"):
    val_dataset = EmbeddingQAFinetuneDataset.from_json(val_dataset_dir)
    bge_val_results = evaluate(val_dataset, embed_model)
    df_bge = pd.DataFrame(bge_val_results)
    hit_rate_bge = df_bge['is_hit'].mean()
    print(f"{model_name}模型的准确率为：{hit_rate_bge}")
    # bge-large-en-v1.5模型的准确率为：0.6155913978494624
    # 微调后的bge-large-en-v1.5模型的准确率为：0.7093023255813954
