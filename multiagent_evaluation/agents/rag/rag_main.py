# python -m rag.rag
# this module encapsulates the RAG (Retrieval Augmented Generation) implementation
# it leverages AIModule class and aisearch module to search for the answer
# create RAG class

from multiagent_evaluation.aisearch.ai_search import AISearch
from multiagent_evaluation.aimodel.ai_model import AIModel
from multiagent_evaluation.session_store.session_store import SimpleInMemorySessionStore
from multiagent_evaluation.utils.utils import configure_tracing, get_credential, configure_logging, load_agent_configuration
import os
import json
# initialize all environment variables from .env file
from opentelemetry import trace
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

from dotenv import load_dotenv
load_dotenv()


# Configure tracing and logging
logger = configure_logging()
tracer = configure_tracing(__file__)


class RAG:

    def __init__(self, rag_config: dict = None) -> None:

        logger.info("RAG.Initializing RAG")
        with tracer.start_as_current_span("RAG.__init__span") as span:
            try:
                if rag_config is None:
                    # load configuration from default variant yaml in the agents/rag folder
                    logger.info(
                        "RAG.__init__#rag_config is empty, loading default configuration")
                    rag_config = load_agent_configuration(
                        "agents/rag", "rag_agent_config.yaml")

                logger.info(f"RAG.__init__#agent_config = {rag_config}")
                span.set_attribute("agent config", rag_config)

                self.api_key = os.getenv("AZURE_OPENAI_KEY")
                # check if ragConfig is not None - throw exception
                if rag_config is None or self.api_key is None:
                    logger.error("agent config and api_key are required")
                    raise ValueError("agent config and api_key are required")

                self.rag_config = rag_config

                # init the AIModel class enveloping a LLM model
                self.aimodel = AIModel(
                    azure_deployment=self.rag_config["AgentConfiguration"]["deployment"]["name"],
                    openai_api_version=self.rag_config["AgentConfiguration"]["deployment"]["openai_api_version"],
                    azure_endpoint=self.rag_config["AgentConfiguration"]["deployment"]["endpoint"],
                    api_key=self.api_key,
                    model_parameters={
                        "temperature": self.rag_config["AgentConfiguration"]["model_parameters"]["temperature"]}
                )
                # init the AISearch class , enveloping the Azure Search retriever
                self.aisearch = AISearch(self.rag_config["AgentConfiguration"]["retrieval"]["deployment"]["name"],
                                         self.rag_config["AgentConfiguration"]["retrieval"]["deployment"]["endpoint"],
                                         self.rag_config["AgentConfiguration"]["retrieval"]["parameters"]["index_name"],
                                         self.rag_config["AgentConfiguration"]["retrieval"]["parameters"]["index_semantic_configuration_name"])

                # initiate the session store
                self._session_store = SimpleInMemorySessionStore()

                # create a prompt template for user intent
                # user intent is concluded from a chat history and the current user question
                self._user_intent_prompt_template = ChatPromptTemplate.from_messages(
                    [
                        ("system",
                         self.rag_config["AgentConfiguration"]["intent_system_prompt"]),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),

                    ]
                )
                # create history aware retriever to build a search query for the user intent
                # for more details see:
                # https://api.python.langchain.com/en/latest/chains/langchain.chains.history_aware_retriever.create_history_aware_retriever.html
                # This method creates a chain that takes conversation history and returns documents.
                # If there is no chat_history, then the input is just passed directly to the retriever.
                # If there is chat_history, then the prompt and LLM will be used to generate a search query.
                # That search query is then passed to the retriever.
                self._history_aware_user_intent_retriever = \
                    create_history_aware_retriever(self.aimodel.llm(),
                                                   self.aisearch.create_retriever(
                        self.rag_config["AgentConfiguration"]["retrieval"]["parameters"]["search_type"],
                        self.rag_config["AgentConfiguration"]["retrieval"]["parameters"]["top_k"]),
                        self._user_intent_prompt_template
                    )

                # prepare final chat chain with history aware retriever
                self._chat_prompt_template = ChatPromptTemplate.from_messages(
                    [
                        ("system",
                         self.rag_config["AgentConfiguration"]["chat_system_prompt"]),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ]
                )

                # Create a chain for passing a list of Documents to a model.
                self._question_answer_chain = \
                    create_stuff_documents_chain(
                        self.aimodel.llm(), self._chat_prompt_template)

            # Create retrieval chain that retrieves documents and then passes them on.
                self._rag_chain = \
                    create_retrieval_chain(
                        self._history_aware_user_intent_retriever, self._question_answer_chain)

                # create a chain with message history automatic handling
                self._conversational_rag_chain = RunnableWithMessageHistory(
                    self._rag_chain,
                    self.get_session_history,
                    input_messages_key="input",
                    history_messages_key="chat_history",
                    output_messages_key="answer",
                )
            except Exception as e:
                logger.exception(f"RAG.__init__#exception= {e}")
                raise e

    def __call__(
        self,
        session_id: str,
        question: str = " "
    ) -> str:
        """>>>RAG Flow entry function."""
        with tracer.start_as_current_span("RAG.__call__span") as span:
            logger.info("RAG.__call__start_chat")
            span.set_attribute("session_id", session_id)

            response = self.chat(session_id, question)

            logger.info(f"RAG.__call__#response= {response}")
            return response

    def get_chat_prompt_template(self):
        return self._chat_prompt_template

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:

        logger.info(f"get_session_history#session_id= {session_id}")
        if session_id not in self._session_store.get_all_sessions_id():
            self._session_store.create_session(session_id)

        return self._session_store.get_session(session_id)

    def chat(self, session_id, question, **kwargs):
        logger.info(f"chat#session_id= {session_id}, question= {question}")
        with tracer.start_as_current_span("RAG.__chat__span") as span:
            try:
                span.set_attribute("session_id", session_id)
                # this paramerter is deperecated
                span.set_attribute(
                    "application_name", self.rag_config["AgentConfiguration"]["application_name"])
                # this paramerter is deperecated
                span.set_attribute(
                    "application_version", self.rag_config["AgentConfiguration"]["application_version"])
                span.set_attribute(
                    "config_version", self.rag_config["AgentConfiguration"]["config_version"])
                span.set_attribute(
                    "endpoint", self.rag_config["AgentConfiguration"]["deployment"]["endpoint"])

                response = self._conversational_rag_chain.invoke({"input": question},
                                                                 config={"configurable": {
                                                                     "session_id": session_id}}
                                                                 )
                logger.info(
                    f"rag_main#session_id#response= {session_id}, response= {response}")

            except Exception as e:
                logger.error(f"rag_main chat#exception= {e}")
                raise e
            logger.info(f"rag_main.chat#response= {response}")
            return response["answer"]


if __name__ == "__main__":
    print("$$$$$$$$$$$$rag_main.py")
    rag = RAG()


""" 
     # Initialize the RAG class and empty history
    import uuid
    rag = RAG()
    session_id = str(uuid.uuid4())
    resp = rag(session_id, "What's Microsoft Fabric?")
    print (f"***response1 = {resp}")
    

    resp = rag.chat(session_id, question="What's Microsoft Fabric Data Factory?")
    print (f"***response1 = {resp}")
    
    resp = rag.chat(session_id, question="List all data sources it supports?")
    print (f"***response2 = {resp}")
    
    resp = rag.chat(session_id, question="Does it support CosmosDB?")
    print (f"***response3 = {resp}")
    
    resp = rag.chat(session_id, question="List all my previous questions.")
    print (f"***response4 = {resp}")

    new_session_id = str(uuid.uuid4())
    resp = rag.chat(new_session_id, question="List all my previous questions.")
    print (f"***response5 = {resp}")

"""
