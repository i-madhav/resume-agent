from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from google.adk.agents import Agent
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.ERROR)


def parse_given_path(pdf_path:str) -> str :
    """Retrieves the content of the pdf on the given path.
    Args:
      pdf_path (str): like ex: "C:/Users/madha/Downloads/MadhavSharmaResume.pdf"
    Returns: string
    """
    pdf_paths = [path.strip() for path in pdf_path.split(',')]
    text = ""
    for pdf in pdf_paths:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        except Exception as e:
            print(f"Error reading {pdf}: {e}")
    print(f"--- Tool: parse_given_path called for path: {pdf_path} ---")
    
    if(text):
         print("I got the text")
         chunks = get_text_chunks(text)
         if(chunks):
             print("I got the chunk")
             vector_store = get_vector_store(chunks)
             if(vector_store):
                 return "document parsed successfully"
    return "unable to parse the document"

# @title Define the get_weather Tool
def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city.

    Args:
        city (str): The name of the city (e.g., "New York", "London", "Tokyo").

    Returns:
        dict: A dictionary containing the weather information.
              Includes a 'status' key ('success' or 'error').
              If 'success', includes a 'report' key with weather details.
              If 'error', includes an 'error_message' key.
    """
    # Best Practice: Log tool execution for easier debugging
    print(f"--- Tool: get_weather called for city: {city} ---")
    city_normalized = city.lower().replace(" ", "") # Basic input normalization
    print("This is cityNormalized")
    print(city_normalized)
    # Mock weather data for simplicity
    mock_weather_db = {
        "newyork": {"status": "success", "report": "The weather in New York is sunny with a temperature of 25°C."},
        "london": {"status": "success", "report": "It's cloudy in London with a temperature of 15°C."},
        "tokyo": {"status": "success", "report": "Tokyo is experiencing light rain and a temperature of 18°C."},
    }

    # Best Practice: Handle potential errors gracefully within the tool
    if city_normalized in mock_weather_db:
        return mock_weather_db[city_normalized]
    else:
        return {"status": "error", "error_message": f"Sorry, I don't have weather information for '{city}'."}

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index")
    return True
    
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def provide_information_about_pdf(user_input:str) -> str:
    print(f"--- Tool: provide_information_about_pdf: {user_input} ---")
    """Retrieves the content of the pdf.
    Args:
      user_input (str): tell me name of the person in the pdf
    Returns:Answer the question in as detailed as possible from the provided context. 
    Make sure to provide all the details. If the answer is not in the provided context just say, 
    "answer is not available in the context", don't provide the wrong answer
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_input)
    context = "\n".join([doc.page_content for doc in docs])
    return context
    
root_agent = Agent(
    name="pdf_qa_agent",
    model="gemini-2.0-flash",
    description=(
       "Answers user questions based strictly on text excerpts provided from a specific PDF document."
    ),
    instruction=(
        "At first ask for path for pdf from the user"
        "After parsing is completed , tell user thsy can ask you a question about pdf"
          "You are a specialized assistant for answering questions about a PDF document. "
        "You will receive relevant text excerpts from the document as context along with the user's question.\n"
        "Your task is to answer the user's question based **exclusively** on the provided context.\n"
        "Follow these rules strictly:\n"
        "1. Base your entire answer **only** on the information contained within the provided text excerpts.\n"
        "2. If the answer cannot be found in the provided context, you **must** state clearly: 'The answer is not found in the provided document context.'\n"
        "3. Do **not** use any prior knowledge, external information, or make assumptions beyond the provided text.\n"
        "4. Answer accurately and directly address the user's question."
    ),
    tools=[parse_given_path,provide_information_about_pdf],
)

session_service = InMemorySessionService()
runner = Runner(
    agent=root_agent,
    app_name="pdf_qa_agent", 
    session_service=session_service
)
