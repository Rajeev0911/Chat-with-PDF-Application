from transformers import pipeline
import os
from dotenv import load_dotenv

load_dotenv()


class LLMHandler:
    def __init__(self):
        self.qa_pipeline = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2"
        )

    def get_answer(self, question: str, context: str) -> str:
        """Get answer from the LLM"""
        result = self.qa_pipeline(
            question=question,
            context=context,
        )
        return result['answer']
