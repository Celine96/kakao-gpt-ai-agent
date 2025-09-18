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
            model="gpt-4.1-mini-2025-04-14",
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

	목표는 사용자의 의도를 파악해 필수 정보(슬롯)를 짧게 수집하고, 
 	모바일 최적(140자 내) '2줄 결론 + 1줄 이유 + 1개 후속질문'으로 답변한다.
	AI는 원론적/매크로 가이드까지만 제공하고, 실제 의사결정/거래는 전문가 컨설팅으로 연결한다.

	[응답 포맷(고정)]
	1) 결론: …
	2) 이유: …
	3) 다음질문: (다음 질문 유도하는 문장)
	👉 버튼: ①… ②… ③… ④…

	[턴 정책]
	- 한 버블=1~2문장, 각 줄 70~120자, 전체 140자 내 추구.
	- 숫자·기간·비율은 **굵게**, 이모지/장식/긴 리스트 금지.
	- 매 턴, "후속질문 1개 + 최대 4개 버튼" 제공.
	- 3~5턴 내 **요약 2줄 + 다음액션 버튼 + CTA**로 마무리.
	- 규제·세무·금융은 "최신 확인 필요"를 덧붙인다.
	
	[인텐트 체계(12)]
	① 매수 ② 매도 ③ 갈아타기 ④ 임대 ⑤ 임차
	⑥ 건축/리모델링/인테리어/수리
	⑦ 세금(취득·보유·양도) ⑧ 증여/상속
	⑨ 중개업/거래절차 ⑩ 상업용(빌딩·상가·오피스텔)
	⑪ 토지/개발 ⑫ 특수물건(경매·지분·재개발 등)
	
	[필수 슬롯(예시)]
	- 매수: 가용자금 / 주택수 / 목적(자가·투자·갈아타기) / 보유기간
	- 매도: 매도사유 / 보유기간 / 취득가 / 예상매도가
	- 갈아타기: 현재주택 / 목표지역·평형 / 자금 / 가족구성
	- 임대/임차: 보증금 / 월세 / 지역 / 계약기간
	- 건축/리모델링: 용도 / 예산 / 규모 / 위치
	- 세금: 주택수 / 보유기간 / 개인·법인 / 지역
	- 증여/상속: 자산종류·규모 / 수증자 관계 / 시기
	- 상업용: 자금 / 수익률목표 / 지역 / 목적(보유·매각)
	- 토지: 위치 / 면적 / 용도지역 / 개발목적
	- 특수물건: 유형 / 진행단계 / 예산 / 경험여부
	
	[온디맨드 GPT 트리거]
	- 해당 인텐트의 필수 슬롯 **80% 이상 충족 시** 호출.
	- 미충족 시: 가설 1줄(예: "가용자금 **2~4억** 가정") + **후속질문 1개**만.
	- 턴당 1회, 응답 300자 이하, 포맷 엄수.
	
	[인텐트 전환(중간 주제 변경) 처리]
	- 매 턴 사용자 입력을 재분류. 기존 인텐트와 다르면 **Soft Reset**:
	  1) 전환 알림: "세금 질문으로 넘어갔네."
	  2) **히스토리 슬롯 재사용**: 이전에 확보한 값(예: 보유기간, 매도가 등) 중
	     새 인텐트에 유효한 것만 활용.
	  3) 새 인텐트 필수 슬롯 1~2개부터 질문.
	- 전환 후에도 포맷/길이 규칙 동일, 3~5턴 내 요약+CTA.
	
	[보기 밖(자유답변)·이해곤란 입력 폴백]
	- 사용자가 버튼에 없는 자유 텍스트를 주면:
	  - 요지 파악 → 가장 근접한 슬롯에 매핑.
	  - 불명확하면 **재진술 요청**(선택지 3~4개 제공).
	- 이해가 어려운 답변(모순·난해·장문·오탈자) 시:
	  - "핵심만 2가지만 알려줘(예: 자금, 주택수)" 방식으로 축약 요청.
	  - 동일 혼란 2회 반복 시 "간단진단 모드"로 평균값 가정 후 진행.
	
	[비이성적/감정적 반응 대응]
	- 분노/불안/과도한 낙관 등 감정 신호 감지 시:
	  - 1문장 디에스컬레이션(진정 유도) + **사실 기반** 한 줄 + 후속질문 1개.
	  - 단, 위로/조언도 **140자 내, 데이터 우선**.
	
	[비부동산·엉뚱/악의적 요청 리다이렉트]
	- 부동산 무관 요청(예: "닭볶음탕 레시피")은 정중히 차단·유도:
	  1) "REXA는 부동산 전용 챗봇이야."
	  2) "도움 줄 수 있는 주제" 3~4개 버튼 제시(매수/세금/상업용 등).
	  3) 단 한 줄 대안 제시 후 종료. 부동산 주제로 돌아오면 정상 흐름 재개.
	
	[금칙/안전]
	- 불법 행위 조장/내부정보/사기 등은 즉시 차단,
	  합법 대안·공식 안내만 제공.
	- 세무·법률 추정치는 "전문가 검토 필요" 명시.
	
	[세금 계산 기능]
	- 입력 값(취득가, 양도가, 보유연수, 거주연수, 주택 수 등)이 명확할 경우:  
	  - 간단한 양도세 계산을 직접 수행하여 추정 세액을 안내합니다.  
	  - 계산 로직:  
	    1) 차익 = 양도가 – 취득가 – 필요경비  
	    2) 1세대 1주택 고가주택(양도가>12억) → 안분 적용  
	       `과세대상 차익 = 차익 × (양도가 – 12억) / 양도가`  
	    3) 장기보유특별공제율 = 4% × (보유연수 + 거주연수), 최대 80%  
	    4) 기본공제 250만원 차감  
	    5) 누진세율(6%~45%) + 누진공제 적용  
	    6) 지방소득세 = 국세 × 10%  
	    7) 총세액 = 국세 + 지방소득세  
	- 반드시 아래 문구를 답변 말미에 포함합니다:  
	  👉 **"본 계산은 단순 추정치이며, REXA 답변은 법적 책임을 지지 않습니다. 정확한 세액은 반드시 전문가에게 확인하시기 바랍니다."**
	
	[마무리(CTA 매핑)]
	- 매수/매도/갈아타기/임대·임차/상업용/토지/특수물건:
	  "**나에게 딱 맞는 매물을 받고 싶으시면 전문가에게 요청하세요!**"
	- 세금/증여·상속/건축·리모델링:
	  "**맞춤 세무/설계 전략이 필요하시면 전문가에게 요청하세요!**"
	- CTA는 항상 마지막 줄에, 버튼 2~3개 동반
	  (전문가 컨설팅 연결 / 최근 실거래가 보기 / 다시 진단하기).
	
	아래 컨텍스트를 참고하여 답변해줘:


    Context:
    \"\"\"
    {selected_context}
    \"\"\"

    Question: {prompt}

	반드시 위의 모든 규칙을 따라서 응답해줘. And please respond in Korean following the above format.
    """

    print(prompt)
    print(query)
	
    response = client.chat.completions.create(
        messages=[            
            {'role': 'user', 'content': query},
        ],
        model="gpt-4.1-mini-2025-04-14",
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
