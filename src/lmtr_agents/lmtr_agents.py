"""Define lmtr_agents."""

import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI

from ycecream import y

from loguru import logger
from lmtr_agents.templates import template_tr, template_ref, template_imp, template_comb

# from lmtr_agents.__init__ import agent_base_url, agent_api_key

agent_model = os.getenv("AGENT_MODEL")
agent_base_url = os.getenv("AGENT_BASE_URL")
agent_api_key = os.getenv("AGENT_API_KEY")

def lmtr_agents():
    """Define lmtr_agents."""
    logger.debug(" entry ")


def agent_tr(
    text: str,
    model: str = agent_model,
    base_url: str = agent_base_url,
    api_key: str = agent_api_key,
    to_lang: str = "Chinese",
    temperature: float = 0.3,
    verbose: bool = False,
) -> str:
    """Translate."""
    prompt_tr = ChatPromptTemplate.from_template(template_tr)

    model_tr = ChatOpenAI(model=model, base_url=base_url, api_key=api_key,temperature=temperature, verbose=verbose)

    chain_tr = prompt_tr | model_tr | StrOutputParser()

    trtext = chain_tr.invoke({"to_lang": to_lang, "text": text})

    return trtext

def agent_ref(
    text: str,
    trtext: str,
    model: str = agent_model,
    base_url: str = agent_base_url,
    api_key: str = agent_api_key,
    to_lang: str = "Chinese",
    temperature: float = 0.3,
) -> str:
    """Reflect initial translation."""
    prompt_ref = ChatPromptTemplate.from_template(template_ref)
    model_ref = ChatOpenAI(model=model, base_url=base_url, api_key=api_key,temperature=temperature, verbose=verbose)
    chain_ref= prompt_ref | model_ref | StrOutputParser()
    edtext = chain_ref.invoke({"text": text, "to_lang": to_lang, "trtext": trtext})

    return edtext

def agent_imp(
    text: str,
    trtext: str,
    reflection: str,
    model: str = agent_model,
    base_url: str = agent_base_url,
    api_key: str = agent_api_key,
    to_lang: str = "Chinese",
    temperature: float = 0.3,
) -> str:
    # improve
    prompt_imp = ChatPromptTemplate.from_template(template_imp)
    model_imp = ChatOpenAI(model=model, base_url=base_url, api_key=api_key,temperature=temperature, verbose=verbose)
    chain_imp = prompt_imp | model_imp | StrOutputParser()
    ftext = chain_imp.invoke({"text": text, "to_lang": to_lang, "trtext": trtext, "reflection": edtext})

    return ftext
