import gradio as gr
from PIL import Image
import os
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatHuggingFace
import asyncio
from gtts import gTTS
import tempfile

load_dotenv()

def load_db(file, chain_type, k):
    async def load_and_process():
        try:
            loader = PyPDFLoader(file)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            docs = text_splitter.split_documents(documents)
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            db = DocArrayInMemorySearch.from_documents(docs, embeddings)
            retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
            qa = ConversationalRetrievalChain.from_llm(
                llm=ChatHuggingFace(model_name="gpt2", temperature=0.7),
                chain_type=chain_type,
                retriever=retriever,
                return_source_documents=True,
                return_generated_question=True,
            )
            return qa
        except Exception as e:
            return f"Erreur lors du chargement de la base de données : {str(e)}"

    return asyncio.run(load_and_process())

async def chat_with_bot(cb, query):
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: cb({"question": query, "chat_history": []}))
        return response
    except Exception as e:
        return {"answer": f"Erreur lors de la communication avec le chatbot : {str(e)}"}

def chatbot_interface(uploaded_file, query):
    if uploaded_file:
        file_path = os.path.join("./", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        cb = load_db(file_path, "stuff", 4)
        result = asyncio.run(chat_with_bot(cb, query))
        response = result.get("answer", "Désolé, je n'ai pas pu répondre à votre question.")
        return response
    else:
        return "Veuillez télécharger un fichier PDF."

def chatbot_interface_with_voice(uploaded_file, audio_query):
    if uploaded_file:
        file_path = os.path.join("./", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        cb = load_db(file_path, "stuff", 4)

        # Convertir l'audio en texte
        query = gr.inputs.Audio().interpret(audio_query)

        # Obtenir la réponse du chatbot
        result = asyncio.run(chat_with_bot(cb, query))
        response_text = result.get("answer", "Désolé, je n'ai pas pu répondre à votre question.")

        # Convertir la réponse texte en audio
        tts = gTTS(response_text, lang='fr')
        with tempfile.NamedTemporaryFile(delete=True) as fp:
            tts.save(fp.name)
            with open(fp.name, "rb") as f:
                response_audio = f.read()
        
        return response_text, response_audio
    else:
        return "Veuillez télécharger un fichier PDF.", None

# Interface Gradio avec fonctionnalité vocale
iface = gr.Interface(
    fn=chatbot_interface_with_voice,
    inputs=[
        gr.inputs.File(label="Télécharger le fichier PDF"),
        gr.inputs.Audio(source="microphone", type="filepath", label="Poser une question (audio)")
    ],
    outputs=[
        "text",
        gr.outputs.Audio(type="auto", label="Réponse audio")
    ],
    title="ANSD's bot",
    description="Un assistant dédié à l'ANSD. Posez des questions sur les informations statistiques du Sénégal."
)

# Lancer l'application
iface.launch()
