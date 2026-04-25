import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

loader = TextLoader("aitopics.txt", encoding="utf-8")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,       
    do_sample=True,           
    temperature=0.7,          
    repetition_penalty=1.5    
)

llm = HuggingFacePipeline(pipeline=pipe)

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""Read the context carefully and give a detailed answer to the question using only the information provided.

Context:
{context}

Question: {question}

Write a complete and detailed answer below:
Answer:"""
)


qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=True   
)

query = input("Ask your question: ")
result = qa_chain.invoke({"query": query})

print("\nAnswer:")
print(result["result"])

print("\nSources used:")
for doc in result["source_documents"]:
    print("-", doc.page_content[:100])  