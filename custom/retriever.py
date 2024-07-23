from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
)
from llama_index.core import QueryBundle
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.schema import NodeWithScore
from typing import List


class CustomRetriever(BaseRetriever):
    """Custom retriever that performs both semantic search and hybrid search."""

    def __init__(
            self,
            vector_retriever: VectorIndexRetriever,
            bm25_retriever: BM25Retriever,
            mode: str = "AND",
            alpha: float = 0.5,
    ) -> None:
        """Init params."""

        self._vector_retriever = vector_retriever
        self._keyword_retriever = bm25_retriever
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode
        self._alpha = alpha
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        keyword_nodes = self._keyword_retriever.retrieve(query_bundle)
        # for node in vector_nodes:
        #     print(node)
        # print("----------")
        # for node in keyword_nodes:
        #     print(node)
        # print("----------")
        vector_ids = {n.node.node_id for n in vector_nodes}
        keyword_ids = {n.node.node_id for n in keyword_nodes}

        # 归一化
        vector_score = [n.score for n in vector_nodes]
        bm25_score = [n.score for n in keyword_nodes]
        minV = min(vector_score)
        maxV = max(vector_score)
        minB = min(bm25_score)
        maxB = max(bm25_score)
        for n in vector_nodes:
            n.score = (n.score - minV) / (maxV - minV)
        for n in keyword_nodes:
            n.score = (n.score - minB) / (maxB - minB)

        # for node in vector_nodes:
        #     print(node)
        # print("----------")
        # for node in keyword_nodes:
        #     print(node)
        # print("----------")

        # 分数加权合并
        combined_dict = {n.node.node_id: n for n in vector_nodes}
        for n in keyword_nodes:
            if n.node.node_id in combined_dict.keys():
                combined_dict[n.node.node_id].score += self._alpha * float(n.score)

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(keyword_ids)
        else:
            retrieve_ids = vector_ids.union(keyword_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]

        for node in retrieve_nodes:
            print(node)
        print("----------")
        return retrieve_nodes
