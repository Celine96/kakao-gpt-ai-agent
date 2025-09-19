import logging
import os

from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI, OpenAIError, APITimeoutError
import numpy as np

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

client = OpenAI(api_key="...")

# Define Pydantic models for nested JSON structure
class DetailParams(BaseModel):
    prompt: dict

class Action(BaseModel):
    params: dict
    detailParams: dict

class RequestBody(BaseModel):
    action: Action

@app.post("/generate")
async def generate_text(request: RequestBody):
    # Extract prompt from nested JSON
    prompt = request.action.params.get("prompt")
    try:
        # Call OpenAI API with the provided prompt
        response = client.responses.create(
            model="gpt-5-mini-2025-08-07",
            input=prompt
        )
        # Return the generated text
        return {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": response.output_text
                        }
                    }
                ]
            }
        }
    except APITimeoutError as e:
        logging.error(f"OpenAI API timeout: {e}")
        return {"error": "OpenAI API timeout occurred."}
    except OpenAIError as e:
        logging.error(f"OpenAI API error: {e}")
        return {"error": "OpenAI API error occurred."}
    except Exception as e:
        logging.error(f"Unknown error: {e}")
        return {"error": "Unknown error occurred."}
## Embeddings

import pickle

with open("embeddings.pkl", "rb") as f:
    data = pickle.load(f)
    article_chunks = data["chunks"]
    chunk_embeddings = data["embeddings"]

# 질문 임베딩

@app.post("/custom")
async def generate_custom(request: RequestBody):
    # Extract prompt from nested JSON
    prompt = request.action.params.get("prompt") # USER INPUT
    q_embedding = client.embeddings.create(input=prompt, model="text-embedding-3-small").data[0].embedding
    
    def cosine_similarity(a, b):
        from numpy import dot
        from numpy.linalg import norm
        return dot(a, b) / (norm(a) * norm(b))

    similarities = [cosine_similarity(q_embedding, emb) for emb in chunk_embeddings]
    
    # 4. 가장 유사한 청크 N개 선택 (여기선 2개)
    top_n = 2
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    selected_context = "\n\n".join([article_chunks[i] for i in top_indices])

    # 5. GPT에게 전달할 메시지 구성
    query = f"""You are a real estate expert advisor for the 'REXA' KakaoTalk chatbot. Please respond to user questions with a polite and trustworthy attitude.

	아래 컨텍스트를 참고하여 답변해줘:

    Context:
    \"\"\"
    {selected_context}
    \"\"\"

    Question: {prompt}

	And please respond in Korean following the above format.
    """

    print(prompt)
    print(query)
	
    response = client.chat.completions.create(
        messages=[            
            {'role': 'user', 'content': query},
        ],
        model="gpt-5-mini-2025-08-07",
        temperature=0,
    )
    
    # Return the generated text
    return {
        "version": "2.0",
        "template": {
        "outputs": [
            {
                "simpleText": {
                    "text": response.choices[0].message.content
                }
            }
        ]
        }                
    }
