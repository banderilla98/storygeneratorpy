from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class StoryOptionLLM(BaseModel):
    text: str = Field(description="the text of the option show to the users")
    nextNode: Dict[str, Any] = Field(description="the next node content and its options")

class StoryNodeLLM(BaseModel):
    content: str = Field(description="The main content of the story node")
    isEnding: bool = Field(description="Whether this node is an ending node")
    isWinningEnding: bool = Field(description="Where this node is a winning ending node")
    options: Optional[List[StoryOptionLLM]] = Field(default=None, description="The options for this node")

class StoryLLResponse(BaseModel):
    title: str = Field(description="The title of the story")
    rootNode: StoryNodeLLM = Field(description="The root node of the story")