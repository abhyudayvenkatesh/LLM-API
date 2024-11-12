from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from constants import openai_key, hf_key
import os
import openai
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
import uvicorn

app = FastAPI()
os.environ['OPENAI_API_KEY']=openai_key
# stt_model = whisper.load_model("base")

# openai.api_key = openai_key

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, user_auth_token=hf_key
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id,user_auth_token=hf_key)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

prompt_template = "Summarize the following conversation:\n{transcription_text}"
prompt = PromptTemplate(input_variables=["transcription_text"], template=prompt_template)
llm = OpenAI(temperature=0.5)  
llm_chain = LLMChain(llm=llm, prompt=prompt)

class TranscriptionResponse(BaseModel):
    text: str
    timestamps: list[dict]

class SummaryResponse(BaseModel):
    summary: str

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(audio: UploadFile = File(...)):
    audio_data = await audio.read()
    
    result = pipe(audio_data)
    
    transcription_text = result["text"]
    timestamps = result['chunks']
    
    return TranscriptionResponse(text=transcription_text, timestamps=timestamps)

@app.post("/summarize", response_model=SummaryResponse)
async def summarize_transcription(transcription_text: str):
    summary = llm_chain.run(transcription_text)
    
    return SummaryResponse(summary=summary)

if __name__=='__main__':
    uvicorn.run(host='127.0.0.1', port=8000, reload=True)
