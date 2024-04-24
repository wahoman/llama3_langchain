import os
from fastapi import FastAPI, Form
from starlette.responses import HTMLResponse, FileResponse
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
import openai
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# 환경 변수 로드
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# OpenAI API 설정
openai.api_key = openai_api_key

# Llama 모델과 토크나이저 로드
model_id = "beomi/Llama-3-Open-Ko-8B-Instruct-preview"
tokenizer = AutoTokenizer.from_pretrained(model_id)
llama_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)

# PDF 문서 로딩 및 텍스트 분할
loader = PyPDFLoader("C:/Users/SSTLabs/Desktop/여형구/20. 장민섭_YOLOv5를 이용한 2D X-Ray 이미지 상의 폭발물 탐지_Rev3.0.pdf")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
vector_store = Chroma.from_documents(texts, embeddings)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return FileResponse('static/llama3.html')

@app.post("/answer/")
async def answer(question: str = Form(...)):
    try:
        # 유사도 기반 검색을 사용
        search_results = vector_store.search(query=question, k=2, search_type='similarity')
        # 각 Document 객체에서 page_content를 통해 텍스트 추출
        context = " ".join([result.page_content for result in search_results])
        input_ids = tokenizer(context + question, return_tensors="pt").input_ids.to(llama_model.device)

        outputs = llama_model.generate(input_ids, max_length=500)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        answer = f"An error occurred: {str(e)}"
        for result in search_results:
            print(dir(result))
    return {"answer": answer}



