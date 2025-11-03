from sqlalchemy.orm import Session
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from core.prompts import STORY_PROMPT
from models.story import Story, StoryNode
from core.models import StoryLLMResponse, StoryNodeLLM
from dotenv import load_dotenv
import os
import requests
load_dotenv()

class StoryGenerator:

    @classmethod
    def _get_llm(cls):
        """
        Returns a ChatOpenAI instance configured for either:
        1. Direct OpenAI API key
        2. Choreo-managed OAuth credentials
        """
        api_key = os.getenv("CHOREO_OPENAI_CONNECTION_OPENAI_API_KEY")
        service_url = os.getenv("CHOREO_OPENAI_CONNECTION_SERVICEURL")

        consumer_key = os.getenv("CHOREO_OPENAI_CONNECTION_CONSUMERKEY")
        consumer_secret = os.getenv("CHOREO_OPENAI_CONNECTION_CONSUMERSECRET")
        token_url = os.getenv("CHOREO_OPENAI_CONNECTION_TOKENURL")

        # 1️⃣ If we have a direct API key and service URL, use it
        if api_key and service_url:
            return ChatOpenAI(model="gpt-4o-mini", api_key=api_key, base_url=service_url)

        # 2️⃣ If Choreo OAuth credentials exist, fetch a token first
        if consumer_key and consumer_secret and token_url:
            token_response = requests.post(
                token_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": consumer_key,
                    "client_secret": consumer_secret,
                },
            )
            token_response.raise_for_status()
            access_token = token_response.json().get("access_token")
            if not access_token:
                raise ValueError("Failed to fetch access token from Choreo OAuth")

            return ChatOpenAI(
                model="gpt-4o-mini",
                api_key=access_token,  # LangChain uses api_key for Bearer token
                base_url=service_url
            )

        # 3️⃣ Default to standard OpenAI if nothing else is set
        return ChatOpenAI(model="gpt-4o-mini")

    @classmethod
    def generate_story(cls, db: Session, session_id: str, theme: str = "fantasy")-> Story:
        llm = cls._get_llm()
        story_parser = PydanticOutputParser(pydantic_object=StoryLLMResponse)

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                STORY_PROMPT
            ),
            (
                "human",
                f"Create the story with this theme: {theme}"
            )
        ]).partial(format_instructions=story_parser.get_format_instructions())

        raw_response = llm.invoke(prompt.invoke({}))

        response_text = raw_response
        if hasattr(raw_response, "content"):
            response_text = raw_response.content

        story_structure = story_parser.parse(response_text)

        story_db = Story(title=story_structure.title, session_id=session_id)
        db.add(story_db)
        db.flush()

        root_node_data = story_structure.rootNode
        if isinstance(root_node_data, dict):
            root_node_data = StoryNodeLLM.model_validate(root_node_data)

        cls._process_story_node(db, story_db.id, root_node_data, is_root=True)

        db.commit()
        return story_db

    @classmethod
    def _process_story_node(cls, db: Session, story_id: int, node_data: StoryNodeLLM, is_root: bool = False) -> StoryNode:
        node = StoryNode(
            story_id=story_id,
            content=node_data.content if hasattr(node_data, "content") else node_data["content"],
            is_root=is_root,
            is_ending=node_data.isEnding if hasattr(node_data, "isEnding") else node_data["isEnding"],
            is_winning_ending=node_data.isWinningEnding if hasattr(node_data, "isWinningEnding") else node_data["isWinningEnding"],
            options=[]
        )
        db.add(node)
        db.flush()

        if not node.is_ending and (hasattr(node_data, "options") and node_data.options):
            options_list = []
            for option_data in node_data.options:
                next_node = option_data.nextNode

                if isinstance(next_node, dict):
                    next_node = StoryNodeLLM.model_validate(next_node)

                child_node = cls._process_story_node(db, story_id, next_node, False)

                options_list.append({
                    "text": option_data.text,
                    "node_id": child_node.id
                })

            node.options = options_list

        db.flush()
        return node