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

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
            model="gpt-5",
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

import pickle #파이썬에서 기본으로 제공하는 라이브러리로, gpt가 이해할 수 있는 형태로 chunck를 가져오는 역할(from embedding2.py에서 저장한 pickle 파일). 

with open("embeddings.pkl", "rb") as f:
    data = pickle.load(f)
    article_chunks = data["chunks"]
    chunk_embeddings = data["embeddings"]

# 질문 임베딩

@app.post("/custom")
async def generate_custom(request: RequestBody):
    # Extract prompt from nested JSON
    prompt = request.action.params.get("prompt") # USER INPUT
    q_embedding = client.embeddings.create(input=prompt, model="text-embedding-3-small").data[0].embedding #embedding pickle 파일을 이해해서 gpt에게 넘기는 역할 
    
    def cosine_similarity(a, b):
        from numpy import dot
        from numpy.linalg import norm
        return dot(a, b) / (norm(a) * norm(b)) #user input과 embedding pickle 파일의 코싸인 유사도 계산 

    similarities = [cosine_similarity(q_embedding, emb) for emb in chunk_embeddings]
    
    # 4. 가장 유사한 청크 N개 선택 (여기선 2개)
    top_n = 2
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    selected_context = "\n\n".join([article_chunks[i] for i in top_indices])

    # 5. GPT에게 전달할 메시지 구성
    query = f"""Use the below context to answer the question. 
	You are REXA, a chatbot that is a real estate expert with 10 years of experience in taxation (capital gains tax, property holding tax, gift/inheritance tax, acquisition tax), auctions, civil law, and building law. 
	Respond politely and with a trustworthy tone, as a professional advisor would. To ensure fast responses, keep your answers under 300 tokens. 
	If you don't know about the information ask the user once more time.
	
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
        model="gpt-5",
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
