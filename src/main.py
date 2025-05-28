import faiss
import asyncio
from sentence_transformers import SentenceTransformer
import os
from glob import glob
from PIL import Image
import numpy as np
from .web_search import google_crawl
from .web_search.faiss_stuff import create_index_with_text, retrieve_relevant_docs
from .web_search.llm import chat

# model = SentenceTransformer('clip-ViT-B-32')

def generate_clip_embeddings(images_path, model):

    image_paths = glob(os.path.join(images_path, '**/*.jpg'), recursive=True) + glob(os.path.join(images_path, '**/*.png'), recursive=True)
    
    embeddings = []
    for img_path in image_paths:
        print("got img")
        image = Image.open(img_path)
        embedding = model.encode(image)
        embeddings.append(embedding)
    
    return embeddings, image_paths

def get_text_embeddings(text, model):
    embedding = model.encode(text)
    return embedding, text

def search_in_index(model):
    img_embeddings, img_paths = generate_clip_embeddings("imgs", model)
    d = len(img_embeddings[0])

    index = faiss.IndexFlatIP(d) # inner product (cosine sim)

    img_embeddings = np.array(img_embeddings, dtype='float32')
    index.add(img_embeddings)
    print(index.ntotal)
    text_embedding, text = get_text_embeddings("a colorful cameleon", model)
    text_embedding = np.array([text_embedding], dtype='float32')
    D, I = index.search(text_embedding, 5)
    print(I)
    for idx in I[0]:
        print(f"close match {img_paths[idx]}")
    print(D)

def chat_loop():
    user_input = ""
    while user_input.lower() != "exit":
        user_input = input("Ask a question to the AI: ")
        result_docs = google_crawl.get_top_results(user_input, 5)
    
        index, text_chunks = create_index_with_text(result_docs)
        retrieved_info = retrieve_relevant_docs(user_input, index, text_chunks, 5)
        asyncio.run(chat(user_input, " ".join(retrieved_info)))

if __name__ == '__main__':
    chat_loop()


    