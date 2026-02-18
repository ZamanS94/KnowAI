from pydantic import BaseModel
from openai import OpenAI
import instructor
from deepeval.models import DeepEvalBaseLLM

class CustomOpenAI(DeepEvalBaseLLM):
    def __init__(self):
        client = OpenAI(api_key="sk-proj")
        
        self.client = instructor.from_openai(
            client,
            mode=instructor.Mode.JSON,
        )

        self.model_name = "gpt-4o-mini"

    def load_model(self):
        return self.client

    def generate(self, prompt: str, schema: BaseModel):
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            response_model=schema,
        )

    async def a_generate(self, prompt: str, schema: BaseModel):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return self.model_name
