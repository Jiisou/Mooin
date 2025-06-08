import gradio as gr
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection
)
import matplotlib.pyplot as plt

# 모델 및 토크나이저 로딩
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text_tokenizer = CLIPTokenizer.from_pretrained("Searchium-ai/clip4clip-webvid150k")
text_model = CLIPTextModelWithProjection.from_pretrained("Searchium-ai/clip4clip-webvid150k").to(device).eval()
vision_model = CLIPVisionModelWithProjection.from_pretrained("Searchium-ai/clip4clip-webvid150k").to(device).eval()

# 전처리 함수
def preprocess(size, image):
    transform = Compose([
        Resize(size, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(size),
        lambda img: img.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])
    return transform(image)

# 영상 → 프레임 이미지 텐서 리스트
def video_to_frames(video_path, frame_rate=1, size=224):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % int(fps / frame_rate) == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            tensor = preprocess(size, img)
            frames.append(tensor)
        count += 1
    cap.release()
    return torch.stack(frames).to(device)

# 텍스트 임베딩 추출
def get_text_embedding(text):
    inputs = text_tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = text_model(**inputs)[0]
        output = output / output.norm(dim=-1, keepdim=True)
    return output

# 영상 임베딩 추출 (프레임 단위)
def get_video_embeddings(frames):
    with torch.no_grad():
        outputs = vision_model(pixel_values=frames)["image_embeds"]
        outputs = outputs / outputs.norm(dim=-1, keepdim=True)
    return outputs

# 유사도 계산 결과 반환
def analyze(video, text):
    # frames = video_to_frames(video.name) ;gr.Video 컴포넌트는 기본값(type="filepath")으로 단순 문자열 경로를 반환
    frames = video_to_frames(video)
    video_embeddings = get_video_embeddings(frames)
    text_embedding = get_text_embedding(text)

    # 유사도 계산
    text_vector = text_embedding.squeeze(0)
    sims = torch.matmul(video_embeddings, text_vector).cpu().numpy()

    # 그래프 객체 생성해 시각화
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(sims, color="crimson", linewidth=2)
    ax.set_title(f"Frame-wise Cosine Similarity to: \"{text}\"")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Similarity")
    ax.grid(True)

    return float(np.mean(sims)), float(np.max(sims)), fig

# Gradio 인터페이스 구성
demo = gr.Interface(

    fn=analyze,
    inputs=[
        gr.Video(label="영상 불러오기"),
        gr.Textbox(label="탐지 상황 설명 입력", placeholder="예: '사람이 누워있다'")
    ],
    outputs=[
        gr.Number(label="📈 평균 유사도"),
        gr.Number(label="📝 최대 유사도"),
        gr.Plot(label="📊 프레임별 유사도 그래프")
    ],
    title="VLM Text–Video 유사도 분석 데모 🎞️",
    description="입력한 텍스트와 영상 프레임 간 유사도를 시각화합니다.",
).launch(share=True)  # share=True 추가하면 공개 링크 생성됨


demo.launch()