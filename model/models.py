from pydantic import BaseModel, RootModel, Field
from typing import List, Union, Optional
from enum import Enum

class Metadata(BaseModel):
    Summary: List[str]
    Title: str
    Author: List[str]
    DateCreated: str
    LastModifiedDate: str
    Publisher: str
    Language: str
    PageCount: Union[int, str]  # Can be "Not Available"
    SentimentTone: str

class ChangeFormat(BaseModel):
    Page: str
    Changes: str

class SummaryResponse(RootModel[list[ChangeFormat]]):
    pass

class DocumentAnswer(BaseModel):
    answer: str = Field(description="Direct answer to the user's question")
    confidence: float = Field(description="Confidence score between 0 and 1")
    sources: List[str] = Field(description="List of source document names or sections")
    reasoning: Optional[str] = Field(description="Brief explanation of the reasoning")
    answer_type: str = Field(description="Type of answer: factual, summary, comparison, etc.")


class PromptType(str, Enum):
    DOCUMENT_ANALYSIS = "document_analysis"
    DOCUMENT_COMPARISON = "document_comparison"
    CONTEXTUALIZE_QUESTION = "contextualize_question"
    CONTEXT_QA = "context_qa"