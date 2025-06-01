from langchain_community.document_loaders import DirectoryLoader
from ragas.llms import LangchainLLMWrapper
from langchain_ollama import ChatOllama
from ragas.testset import TestsetGenerator
from sentence_transformers import SentenceTransformer
from ragas.testset.graph import Node, NodeType, KnowledgeGraph
from ragas.testset.transforms import apply_transforms, default_transforms
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.testset.persona import Persona
from ragas.testset.synthesizers import default_query_distribution
import os




# llm = ChatOllama(model="llama3.2", temperature=0)

# generator_llm = LangchainLLMWrapper(llm)
# embeddings_model_path="Alibaba-NLP/gte-multilingual-base"
# embeddings_model = SentenceTransformer(embeddings_model_path, trust_remote_code=True)

generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

docs_path = "assets/langgraph/docs/docs/agents"
loader = DirectoryLoader(docs_path, glob="**/*.md")
docs = loader.load()
kg = KnowledgeGraph()
for doc in docs:
    kg.nodes.append(
        Node(
            type=NodeType.DOCUMENT,
            properties={"page_content": doc.page_content, "document_metadata": doc.metadata}
        )
    )

'''
This function defines a series of transformation steps to be applied to a knowledge graph, 
including extracting summaries, keyphrases, titles, headlines, and embeddings, 
as well as building similarity relationships between nodes.
'''
transforms = default_transforms(docs, llm=generator_llm, embedding_model=generator_embeddings)

apply_transforms(kg, transforms=transforms)
kg.save("knowledge_graph.json")
#loaded_kg = KnowledgeGraph.load("knowledge_graph.json")


'''
[
    (SingleHopSpecificQuerySynthesizer(llm=llm), 0.5),
    (MultiHopAbstractQuerySynthesizer(llm=llm), 0.25),
    (MultiHopSpecificQuerySynthesizer(llm=llm), 0.25),
]
'''
query_distibution = default_query_distribution(llm=generator_llm, kg=kg)


personas = [
    Persona(
        name="curious engineer",
        role_description="A software engineer who wants to learn how to use LangGraph in a professional setting.",
    ),
    Persona(
        name="novice developer",
        role_description="A developer new to graph-based frameworks who needs clear, step-by-step guidance on LangGraph basics.",
    ),
    Persona(
        name="researcher",
        role_description="An academic or analyst exploring LangGraph’s underlying algorithms and advanced capabilities for a publication or study.",
    ),
    Persona(
        name="system architect",
        role_description="An architect assessing how LangGraph can fit into large-scale system designs and interoperability with existing infrastructure.",
    ),
]


generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings, knowledge_graph=kg, persona_list=personas)
testset = generator.generate(testset_size=50, query_distribution=query_distibution)

testset = testset.to_pandas()
print(testset)
json_path = os.path.join("default_testset.json")
testset.to_json(json_path, orient="records", lines=True)