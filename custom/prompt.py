qa_prompt_tmpl_str = """
Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {query_str}
Answer: 
"""

# 根据如下description选择是否需要进行检索增强生成
simple_qa_prompt_tmpl_str = """
Please answer the query.
Query: {query_str}
Answer: 
"""
rag_description = "Useful for answering questions that require specific contextual knowledge to be answered accurately."
norag_rag_description = ("Used to answer questions that do not require specific contextual knowledge to be answered "
                         "accurately.")