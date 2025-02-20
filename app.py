import os
from flask import Flask, render_template, request, jsonify
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

app = Flask(__name__)

# --- LLM, Database, and QA Chain Setup ---

def initial_llm():
    # Get API key from environment variables
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")
    
    # Initialize ChatGroq with API key and model name
    llm = ChatGroq(
        temperature=0,
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile"
    )
    return llm

def create_db():
    # Use absolute path for PDF file
    pdf_path = os.path.join(os.getcwd(), "The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"🚨 PDF file not found at: {pdf_path}")
    
    # Create vector database
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Use absolute path for ChromaDB
    db_path = os.path.join(os.getcwd(), "chroma_db")
    vector_db = Chroma.from_documents(texts, embeddings, persist_directory=db_path)
    print("✅ ChromaDB created and medical data saved!")
    return vector_db

def setup_qachain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt_template = """You are DiagnoBot, an empathetic medical assistant focused on providing evidence-based health information. Your responses should:
1. Be clear, concise, and use simple language with medical terms in parentheses when needed
2. Always emphasize the importance of consulting healthcare professionals
3. Mark urgent symptoms with ⚠️
4. Provide practical lifestyle and preventive care recommendations
5. Include reliable sources when possible
6. Stay within your scope: no diagnoses, prescriptions, or definitive medical advice
7. Maintain HIPAA compliance and medical ethics
8. Show empathy while remaining professional
9. Keep the answers, diet plans India oriented, cause most of the users will be Indians.
10. Provide mental health related guidelines as well, citing proper references.
11. If users ask questions in bengali, respond them in bengali. If they ask questions in english, respond them in english.

Context from medical resources:
{context}
Question: {question}
Response (structured with clear headings and bullet points when appropriate):"""
    
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

# Initialize vector database and chain
db_path = os.path.join(os.getcwd(), "chroma_db")
if not os.path.exists(db_path):
    vector_db = create_db()
else:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)

llm = initial_llm()
qa_chain = setup_qachain(vector_db, llm)

# --- Flask Routes ---

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"response": "Please enter a valid message."})
    
    try:
        response = qa_chain.invoke({"query": user_message})
        return jsonify({"response": response["result"]})
    except Exception as e:
        return jsonify({"response": f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
