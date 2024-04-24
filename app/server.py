import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Union
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

chain = ChatOpenAI(name="테디")

class HumanMessage(BaseModel):
    text: str

class AIMessage(BaseModel):
    text: str

class SystemMessage(BaseModel):
    text: str

class InputChat(BaseModel):
    """Input for the chat endpoint."""
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]] = Field(
        ...,
        description="The chat messages representing the current conversation.",
    )
    
message_text = "너는 누구인가요"


@app.post("/chat")
def chat_endpoint(input_chat: InputChat):
    # 입력 받은 대화 데이터를 사용하여 답변 생성
    try:
        user_message = input_chat.messages[-1].text  # 사용자의 최근 메시지
        ai_response = list(chain.invoke(user_message))
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return JSONResponse(status_code=500, content={"message": "An error occurred"})
    
    response_data = {"user_message": user_message, "ai_response": ai_response[0][1]}  # 사용자 메시지와 AI 응답 함께 반환
    return JSONResponse(content=response_data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    