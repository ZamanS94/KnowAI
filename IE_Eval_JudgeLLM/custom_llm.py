from pydantic import BaseModel
import google.generativeai as genai
import instructor
from deepeval.models import DeepEvalBaseLLM

class CustomGeminiFlash(DeepEvalBaseLLM):
    def __init__(self):
        genai.configure(api_key="")
        self.model = genai.GenerativeModel(model_name="models/gemini-3-flash-preview")

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: BaseModel):
        client = self.load_model()
        instructor_client = instructor.from_gemini(
            client=client,
            mode=instructor.Mode.GEMINI_JSON,
        )
        resp = instructor_client.messages.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            response_model=schema,
        )
        return resp

    async def a_generate(self, prompt: str, schema: BaseModel):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "Gemini 1.5 Flash"