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