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
from llama_index.core import Settings
from llama_index.core.postprocessor import LLMRerank
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker


def build_query_engine(index: VectorStoreIndex,
                       response_mode: ResponseMode = ResponseMode.TREE_SUMMARIZE,
                       qa_prompt_tmpl: Optional[BasePromptTemplate] = None,
                       with_hybrid_search: bool = False,
                       top_k: int = 5,
                       top_k_rerank: int = 2,
                       with_rerank: bool = True,
                       nodes: Optional[List[BaseNode]] = None):
    reranker = FlagEmbeddingReranker(
        top_n=top_k_rerank,
        model="BAAI/bge-reranker-large",
        use_fp16=True,
    )
    if with_hybrid_search:
        if with_rerank:
            rag_query_engine = index.as_query_engine(similarity_top_k=top_k,
                                                     text_qa_template=qa_prompt_tmpl,
                                                     node_postprocessors=[reranker, MetadataReplacementPostProcessor(
                                                         target_metadata_key="window")],
                                                     sparse_top_k=12,
                                                     vector_store_query_mode="hybrid",
                                                     response_synthesizer=get_response_synthesizer(
                                                         response_mode=response_mode,
                                                         # refine_template=PromptTemplate(refine_tmpl_str)
                                                     ),
                                                     )
        else:
            rag_query_engine = index.as_query_engine(similarity_top_k=top_k,
                                                     text_qa_template=qa_prompt_tmpl,
                                                     node_postprocessors=[MetadataReplacementPostProcessor(
                                                         target_metadata_key="window")],
                                                     sparse_top_k=12,
                                                     vector_store_query_mode="hybrid",
                                                     response_synthesizer=get_response_synthesizer(
                                                         response_mode=response_mode,
                                                         # refine_template=PromptTemplate(refine_tmpl_str)
                                                     ),
                                                     )

    else:
        # Build a tree index over the set of candidate nodes, with a summary prompt seeded with the query. with LLM reranker
        if with_rerank:
            rag_query_engine = index.as_query_engine(similarity_top_k=top_k,
                                                     text_qa_template=qa_prompt_tmpl,
                                                     node_postprocessors=[
                                                         reranker,
                                                         MetadataReplacementPostProcessor(target_metadata_key="window"),
                                                     ],
                                                     response_synthesizer=get_response_synthesizer(
                                                         response_mode=response_mode),
                                                     )
        else:
            rag_query_engine = index.as_query_engine(similarity_top_k=top_k,
                                                     text_qa_template=qa_prompt_tmpl,
                                                     node_postprocessors=[
                                                         MetadataReplacementPostProcessor(target_metadata_key="window"),
                                                     ],
                                                     response_synthesizer=get_response_synthesizer(
                                                         response_mode=response_mode),
                                                     )
    return rag_query_engine
