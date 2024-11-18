from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import os
import openai
from moviepy.editor import VideoFileClip
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
# from constants import openai_key, hf_key
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from fpdf import FPDF
import numpy as np
import uvicorn

app = FastAPI()
# openai.api_key = openai_key
# os.environ['OPENAI_API_KEY']=openai_key
openai_key = os.getenv("openai_key")
hf_key = os.getenv("hf_key")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device
)

prompt_template = "Summarize the following conversation:\n{transcription_text}"
prompt = PromptTemplate(input_variables=["transcription_text"], template=prompt_template)
llm = OpenAI(temperature=0.5, openai_api_key=openai_key)  
llm_chain = LLMChain(llm=llm, prompt=prompt)

class TranscriptionResponse(BaseModel):
    text: str
    timestamps: list[dict]

class SummaryResponse(BaseModel):
    summary: str

def convert_video_to_audio(video_data):
    with open("temp_video.mp4", "wb") as f:
        f.write(video_data)
    clip = VideoFileClip("temp_video.mp4")
    clip.audio.write_audiofile("temp_audio.wav")
    # audio_data = open("temp_audio.wav", "rb").read()
    return convert_audio_to_mono("temp_audio.wav")

def convert_audio_to_mono(audio_path):
    from moviepy.editor import AudioFileClip
    audio_clip = AudioFileClip(audio_path)
    audio_data = audio_clip.to_soundarray(fps=16000)
    if audio_data.ndim == 2:
        audio_data = np.mean(audio_data, axis=1)
    return audio_data


def clean_up_temp_files():
    try:
        os.remove("temp_video.mp4")
        os.remove("temp_audio.wav")
    except FileNotFoundError:
        pass  


def generate_pdf(transcription_text, summary_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Meeting Summary", ln=True, align="C")
    pdf.multi_cell(0, 10, txt=summary_text)

    pdf.cell(200, 10, txt="Full Transcription", ln=True, align="C")
    pdf.multi_cell(0, 10, txt=transcription_text)

    pdf.output("meeting_summary.pdf")
    return "meeting_summary.pdf"

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_video(video: UploadFile = File(...)):
    video_data = await video.read()
    
    audio_data = convert_video_to_audio(video_data)
    

    result = pipe(audio_data, generate_kwargs={'language':'en'},return_timestamps=True)
    transcription_text = result["text"]
    timestamps = result['chunks']
    
    return TranscriptionResponse(text=transcription_text, timestamps=timestamps)

@app.post("/summarize", response_model=SummaryResponse)
async def summarize_transcription(transcription_text: str):
    summary = llm_chain.run(transcription_text)
    return SummaryResponse(summary=summary)

# @app.post("/generate_pdf")
# async def generate_pdf_file(video: UploadFile = File(...)):
#     transcription_response = await transcribe_video(video)
#     transcription_text = transcription_response.text

#     summary_response = await summarize_transcription(transcription_text)
#     summary_text = summary_response.summary

#     pdf_path = generate_pdf(transcription_text, summary_text)
#     return {"pdf_path": pdf_path}
@app.post("/generate_pdf")
async def generate_pdf_file(video: UploadFile = File(...)):
    print("Starting PDF generation process...")
    
    transcription_response = await transcribe_video(video)
    print("Transcription completed:", transcription_response)
    
    transcription_text = transcription_response.text
    summary_response = await summarize_transcription(transcription_text)
    print("Summary completed:", summary_response)
    
    summary_text = summary_response.summary
    pdf_path = generate_pdf(transcription_text, summary_text)
    print("PDF generated:", pdf_path)
    
    result=FileResponse(pdf_path, media_type="application/pdf", filename="meeting_summary.pdf")

    clean_up_temp_files()

    return result

# if __name__=='__main__':
#     uvicorn.run(app, host='127.0.0.1', port=8000, reload=True)
