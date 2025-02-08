import os
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader, UnstructuredExcelLoader
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from dotenv import load_dotenv
from huggingface_hub import login

# Step 2: Set up Hugging Face authentication, model, and tokenizer
def setup_huggingface_model():
    # Load environment variables for Hugging Face token if needed
    load_dotenv()

    # Hugging Face Token for model access
    hf_token = os.getenv("HUGGINGFACE_TOKEN")

    # Log in to Hugging Face
    if hf_token:
        login(token=hf_token)

# Initialize Mistral model
mistral_model_name = 'mistralai/Mistral-7B-Instruct-v0.3'
tokenizer = AutoTokenizer.from_pretrained(mistral_model_name)
mistral_model = AutoModelForCausalLM.from_pretrained(
    mistral_model_name,
    device_map="auto"  # Automatically map to available devices (CPU/GPU)
)

# Create a Hugging Face pipeline
llm_pipeline = pipeline(
    "text-generation", 
    model=mistral_model, 
    tokenizer=tokenizer, 
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.7
)

# Wrap the pipeline with LangChain's HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=llm_pipeline)

# Initialize HuggingFaceEmbeddings for embeddings
embedding_model = HuggingFaceEmbeddings(model_name="paraphrase-mpnet-base-v2")

# Directory for Chroma DB persistence
persist_directory = "/Folder/Chroma_DB"

# Load and process both PDF and Excel files from the directory
folder_path = 'YOUR_FILE_PATH'
pdf_loader = DirectoryLoader(folder_path, glob="*.pdf", loader_cls=PyMuPDFLoader)
excel_loader = DirectoryLoader(folder_path, glob="*.xlsx", loader_cls=UnstructuredExcelLoader)

# Load documents and filter out empty documents
pdf_documents = pdf_loader.load()
excel_documents = excel_loader.load()
documents = pdf_documents + excel_documents
documents = [doc for doc in documents if doc.page_content.strip()]  # Remove empty documents

# Ensure documents are not empty
if not documents:
    raise ValueError("No valid PDF or Excel documents found in the directory.")

# Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# Ensure texts are not empty after splitting
if not texts:
    raise ValueError("Text splitting failed. No content available for processing.")

# Create Chroma DB using `Chroma.from_documents` with HuggingFaceEmbeddings
vectordb = Chroma.from_documents(
    documents=texts,
    embedding=embedding_model,
    persist_directory=persist_directory
)

# Persist the database to disk
vectordb.persist()
vectordb = None

# Reload the persisted database using `Chroma`
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model
)

# Create a retriever
retriever = vectordb.as_retriever()

# Define the QA Chain using the wrapped Hugging Face pipeline
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False
)

# Add logging for debugging
print("Starting query processing...")

# Function to process LLM response and filter for the required output
def process_llm_response(llm_response):
    if not llm_response or 'result' not in llm_response:
        print("I don't know")
        return

    response = llm_response['result']
    # Extract only the relevant part from the response
    if "Helpful Answer:" in response:
        helpful_answer_index = response.find("Helpful Answer:")
        filtered_response = response[helpful_answer_index + len("Helpful Answer:"):].strip()
        if "I don't know" not in filtered_response:
            print("Response:", filtered_response)
        else:
            print("I don't know")
    else:
        print("I don't know")

# User query loop
while True:
    query = input("Enter your query (type 'exit' to stop): ")
    if query.lower() == "exit":
        print("Exiting the program. Goodbye!")
        break
    
    try:
        # Invoke the QA chain with the user query
        llm_response = qa_chain.invoke({"query": query})
        
        # Process the response to extract the relevant helpful answer
        process_llm_response(llm_response)
        
    except Exception as e:
        print(f"An error occurred: {e}")
