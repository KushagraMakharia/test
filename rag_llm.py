from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import GPT4All
from langchain_core.chat_history import BaseChatMessageHistory

from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.prompts import MessagesPlaceholder

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# from langchain.retrievers import ContextualCompressionRetriever
# from langchain.retrievers.document_compressors import CrossEncoderReranker
# from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


class RagLLM:
    def __init__(self, file_path) -> None:
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        self.embeddings = GPT4AllEmbeddings()
        db = FAISS.from_documents(pages, self.embeddings)
        self.retriever = db.as_retriever()
        self.model = GPT4All(model="E:\GPT4ALL\Phi-3-mini-4k-instruct.Q4_0.gguf", n_threads=8)
        self.history = {}
        self.contextual_retriever = None
        self.qa_chain = None
        self.rag_chain = None

    def maintain_history(self, session_id):
        if session_id not in self.history:
            self.history[session_id] = ChatMessageHistory()
        return self.history[session_id]
    
    def create_history_aware_retriever(self):
        contextual_query = """ Given the query and the message history, create a new query which can be interpreted without the chat history.\
Do not add any extra information other than which is available in the chat history and existing query.\
The reultant query must not exceed 50 words.
"""

        contextual_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextual_query),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        self.contextual_retriever = create_history_aware_retriever(self.model, self.retriever, contextual_prompt)

    def invoke_contextual_retriver(self, text):
        if self.contextual_retriever:
            return self.contextual_retriever.invoke({"input": text})
    
    def create_qa_chain(self):
        qa_system_prompt = """ Given the query and its context, create a short and concise response. \
 context: {context}\
"""
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        self.create_history_aware_retriever()
        self.qa_chain = create_stuff_documents_chain(self.model, qa_prompt)
        self.rag_chain = create_retrieval_chain(self.contextual_retriever, self.qa_chain)
        self.conversational_rag_chain = RunnableWithMessageHistory(
                                self.rag_chain,
                                self.maintain_history,
                                input_messages_key="input",
                                history_messages_key="chat_history",
                                output_messages_key="answer"
                            )
        
    async def invoke(self, text):
        async for events in self.conversational_rag_chain.astream_events(
        {"input": text},
                config={
                    "configurable": {"session_id": "4"}
                }, version="v1"
            ):
                kind = events["event"]
                if kind == 'on_llm_stream':
                    data_chunk = events['data']['chunk']
                    yield data_chunk
                    # if len(data_chunk.strip()) > 0:
                    # gen_response.append(data_chunk)
                    # print("".join(gen_response))