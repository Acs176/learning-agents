import json
from ragas import SingleTurnSample
from ..web_search.faiss_stuff import create_index_with_text, retrieve_relevant_docs
from langchain_community.document_loaders import DirectoryLoader
from langchain_ollama import ChatOllama
from ragas.metrics import LLMContextPrecisionWithReference
from ragas.llms import LangchainLLMWrapper
from ragas.evaluation import evaluate
from ragas import EvaluationDataset
from langchain.callbacks.stdout import StdOutCallbackHandler


corpus_path = "assets/langgraph/docs/docs/agents"
testset_path = "default_testset.json"

def build_index(path: str):
    loader = DirectoryLoader(path, glob="**/*.md")
    docs = loader.load()
    docs_str = []
    for doc in docs:
        docs_str.append(doc.page_content)

    index, chunks = create_index_with_text(docs_str)
    return index, chunks

def load_test_set(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Optionally validate keys here
            rows.append(obj)
    return rows


def main():
    testset = load_test_set(testset_path)

    index, chunks = build_index(corpus_path)
    samples = []
    for test in testset:
        top_docs = retrieve_relevant_docs(test["user_input"], index, chunks, 4)
        samples.append(SingleTurnSample(user_input=test["user_input"], retrieved_contexts=top_docs, reference_contexts=test["reference_contexts"], reference=test["reference"]))

    llm = ChatOllama(model="llama3.2", temperature=0)

    generator_llm = LangchainLLMWrapper(llm)
    context_precision = LLMContextPrecisionWithReference(llm=generator_llm)
    evaluation_dataset = EvaluationDataset(samples=samples)
    callbacks = [StdOutCallbackHandler()]

    scores = evaluate(dataset=evaluation_dataset, metrics=[context_precision], callbacks=callbacks)
    print(scores)

if __name__ == "__main__":
    main()