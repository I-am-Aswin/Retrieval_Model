from sklearn.metrics.pairwise import cosine_similarity

def g_score(query_embedding, node_embedding):
    """
    Calculate the similarity (g-score) between query and node.
    """
    return cosine_similarity([query_embedding], [node_embedding])[0][0]

def h_score(query_embedding, node_embedding):
    """
    Calculate heuristic score (h-score) as inverse similarity.
    """
    return 1 - g_score(query_embedding, node_embedding)

def a_star_search(query_embedding, embeddings, sentences, top_k=3):
    """
    Perform A* search for top-k relevant results.
    """
    import heapq

    open_set = []
    results = []

    for idx, node_embedding in enumerate(embeddings):
        g = g_score(query_embedding, node_embedding)
        h = h_score(query_embedding, node_embedding)
        f = g + h
        heapq.heappush(open_set, (-f, idx))

    for _ in range(top_k):
        if not open_set:
            break
        _, idx = heapq.heappop(open_set)
        results.append(sentences[idx])

    return results
