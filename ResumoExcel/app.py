import streamlit as st
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from dotenv import load_dotenv, find_dotenv
import os

# Carregar variáveis de ambiente
_ = load_dotenv(find_dotenv())

# Configurações do modelo da OpenAI
def initialize_model():
    return ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0.7)

# Função para processar o arquivo Excel ou CSV e extrair o texto
def process_file(file_path):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)  # Ler CSV
    else:  # Ler arquivos Excel (xls ou xlsx)
        df = pd.read_excel(file_path)
    return df.to_string(index=False)

# Função para criar uma lista de documentos a partir do texto
def create_documents(text, source):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Ajuste conforme necessário
        chunk_overlap=100  # Para garantir que o contexto seja mantido entre as partes
    )
    text_chunks = splitter.split_text(text)
    documents = [Document(page_content=chunk, metadata={"source": source}) for chunk in text_chunks]
    return documents

# Função para inicializar o vetor de recuperação
def create_vector_store(documents):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

# Função para gerar o resumo e explicação por tópicos usando RAG
def generate_summary_and_explanation(query, vector_store, model):
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    response = qa_chain({"query": query})
    return response["result"], response["source_documents"]

# Interface do Streamlit
def main():
    st.title("Analisador de Arquivos Excel e CSV - Resumo e Explicação ")
    st.write("Faça o upload de um ou mais arquivos Excel (.xlsx, .xls) ou CSV (.csv) para obter um resumo e uma explicação dos principais pontos.")

    uploaded_files = st.file_uploader("Adicione seus arquivos Excel ou CSV", type=["xlsx", "xls", "csv"], accept_multiple_files=True)

    if uploaded_files:
        # Certificar-se de que o diretório "temp" exista
        temp_dir = "temp"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        all_documents = []

        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)

            try:
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            except FileNotFoundError as e:
                st.error(f"Erro ao tentar salvar o arquivo: {e}")
                continue
            
            # Processar o arquivo (Excel ou CSV)
            with st.spinner(f'Processando {uploaded_file.name}...'):
                try:
                    file_text = process_file(file_path)  # Função adaptada para csv/xls/xlsx
                except Exception as e:
                    st.error(f"Erro ao processar o arquivo {uploaded_file.name}: {e}")
                    continue

                if file_text.strip():
                    documents = create_documents(file_text, uploaded_file.name)
                    all_documents.extend(documents)
                else:
                    st.error(f"O arquivo {uploaded_file.name} não contém dados utilizáveis.")

        if all_documents:
            # Criar o vetor de recuperação
            vector_store = create_vector_store(all_documents)

            # Inicializar o modelo
            model = initialize_model()

            # Perguntas do usuário
            queries = st.text_area("Digite suas perguntas ou solicitações de resumo, separadas por linha:")

            # Adiciona um botão de envio
            if st.button("Enviar"):
                if queries:
                    query_list = queries.split("\n")

                    all_results = []
                    download_content = ""

                    for query in query_list:
                        if query.strip():
                            summary_and_explanation, source_documents = generate_summary_and_explanation(query, vector_store, model)

                            # Exibir o resultado para cada pergunta
                            st.markdown(f"### Resumo e Explicação para: '{query}'")
                            st.markdown(summary_and_explanation)

                            # Adicionar as fontes (sources) dos documentos usados na resposta
                            sources = "\n".join([f"- {doc.metadata['source']}" for doc in source_documents])
                            st.markdown(f"**Fontes Utilizadas:**\n{sources}")

                            # Para baixar o conteúdo em .txt
                            download_content += f"### Resumo e Explicação para: '{query}'\n"
                            download_content += summary_and_explanation + "\n"
                            download_content += f"**Fontes Utilizadas:**\n{sources}\n\n"

                    # Botão para baixar o resumo e a explicação como .txt
                    st.download_button(
                        label="Baixar Resumo e Explicação",
                        data=download_content,
                        file_name="resumo_e_explicacao.txt",
                        mime="text/plain"
                    )

            # Remover os arquivos temporários após o processamento
            for uploaded_file in uploaded_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                if os.path.exists(file_path):
                    os.remove(file_path)

if __name__ == "__main__":
    main()
