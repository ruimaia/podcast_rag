from sentence_transformers import util
import torch

def retrieve_k_relevant_resources(query_emb, embeddings, k=3):
    """
    Returns top k most similar embeddings with query.
    """

    dot_scores = util.dot_score(query_emb, embeddings)[0]
    scores, indices = torch.topk(input=dot_scores, k=k)

    return scores, indices


def prompt_formatter(query, context_items):
    context = "- " + "\n- ".join([item for item in context_items]) 

    base_prompt = """Based on the following context items, please answer the query.
    Context items:
    {context}
    Query: {query}
    Answer:
    """

    prompt = base_prompt.format(context=context, query=query)
    print(prompt)

    return prompt