import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import nltk



model_name_or_path="Alibaba-NLP/gte-multilingual-base"
model = SentenceTransformer(model_name_or_path, trust_remote_code=True)
nltk.download("punkt")

def chunk_doc(doc, max_tokens=500, overlap=50):
    '''
    Common practice to keep max_tokens between 400 and 600 (peak performance for most models)
    Overlap is also common to keep between 10 and 20%
    We'll be using a sentence aware splitting method
    '''
    if overlap == None:
        overlap = int(max_tokens / 10)
    sentences = nltk.sent_tokenize(doc)
    chunks = []
    current_chunk = ""
    current_tokens = 0
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    for sent in sentences:
        sent_tokens = len(tokenizer.encode(sent))
        if current_tokens + sent_tokens > max_tokens:
            chunks.append(current_chunk)
            current_chunk = current_chunk[-overlap:] if overlap > 0 else []
            current_tokens = len(tokenizer.encode(current_chunk))
        current_chunk = " ".join([current_chunk, sent])
        current_tokens += sent_tokens

    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def get_embeddings(model, docs, chunk_size=500, overlap=None, is_query=False):
    if not is_query:
        total_chunks = []
        for doc in docs:
            doc_chunks = chunk_doc(doc, chunk_size, overlap)
            total_chunks.extend(doc_chunks)
        embeddings = model.encode(total_chunks, normalize_embeddings=True)
    else:
        total_chunks = None
        embeddings = model.encode([docs], normalize_embeddinds=True)
    return np.array(embeddings, dtype='float32'), total_chunks

def create_index_with_text(inputs_list, chunk_size=500, overlap=None):
    """
    inside here we will split the texts, embed the splits and store them in an index,
    which will be stored as a file
    """
    embeddings, chunks = get_embeddings(model, inputs_list, chunk_size, overlap)
    d = embeddings.shape[1] ## (num_instances, length of emb. vector)
    ids = np.arange(len(embeddings), dtype='int64')
    index = faiss.IndexFlatIP(d)
    index = faiss.IndexIDMap(index)
    index.add_with_ids(embeddings, ids)
    faiss.write_index(index, "corpus.index")
    return index, chunks

def retrieve_relevant_docs(query, index, docs, k):
    query_embeddings, _ = get_embeddings(model, query, is_query=True)
    _, I = index.search(query_embeddings, k)
    print(I)
    top_ids = I[0]
    top_docs = [docs[i] for i in top_ids]
    return top_docs