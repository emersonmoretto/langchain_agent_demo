import openai
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage

import requests
from bs4 import BeautifulSoup
import os

###### Tools

@tool
def llm_tool(text: str, question: str) -> str:
    """ Essa ferramenta recebe um texto de alguma noticia e uma pergunta do usuario sobre o conteúdo dessa noticia """
         
    messages = [
        SystemMessage(content="Você é um analista de sentimento de texto."),
        HumanMessage(content=f"Texto: {text} \n\n {question}")
    ]

    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    response = llm.invoke(messages)
    return response
    

@tool
def read_url(url: str) -> str:
    """ Essa ferramenta recebe uma URL como parametro e captura o texto de uma notícia a partir dessa URL"""

    response = requests.get(url)
    html_content = response.text

    soup = BeautifulSoup(html_content, 'html.parser')

    for script_or_style in soup(["script","style"]):
        script_or_style.decompose()

    all_text = soup.get_text()

    return all_text



####### Agent

toolkit = [llm_tool,read_url]

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
         O usuário fará algumas perguntas sobre o conteúdo da URL. 
         Use suas ferramentas para responder as questões.
         Se voce nao tiver a ferramenta para responder a questao responda Não tenho ferramentas.         
         Retorne apenas as respostas."""
         ),
         MessagesPlaceholder("chat_history",optional=True),
         ("human", "{input}"),
         MessagesPlaceholder("agent_scratchpad")
    ]
)

agent = create_openai_tools_agent(llm, toolkit, prompt)

agent_executor = AgentExecutor(agent=agent, tools=toolkit, verbose=True)

result = agent_executor.invoke({"input":"Me faça uma análise de sentimento dessa notícia https://oglobo.globo.com/politica/noticia/2025/08/05/lula-chora-em-evento-no-planalto-e-da-indireta-a-aliados-de-bolsonaro-se-eu-sair-e-entrar-uma-coisa-a-fome-volta.ghtml"})

print(result["output"])