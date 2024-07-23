from typing import List, Optional
from llama_index.core.schema import BaseNode
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import VectorIndexRetriever
import Stemmer
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from custom.retriever import CustomRetriever
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers.type import ResponseMode


def build_query_engine(index: VectorStoreIndex,
                       response_mode: ResponseMode = ResponseMode.TREE_SUMMARIZE,
                       qa_prompt_tmpl: Optional[BasePromptTemplate] = None,
                       hybrid_search: bool = False,
                       top_k: int = 2,
                       nodes: Optional[List[BaseNode]] = None, ):
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
    return rag_query_engine
