# TODO: Sparse Frame Inference
# TODO: Model Performance Monitoring
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection
import time

# 모델 및 장치
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text_tokenizer = CLIPTokenizer.from_pretrained("Searchium-ai/clip4clip-webvid150k")
text_model = CLIPTextModelWithProjection.from_pretrained("Searchium-ai/clip4clip-webvid150k").to(device).eval()
vision_model = CLIPVisionModelWithProjection.from_pretrained("Searchium-ai/clip4clip-webvid150k").to(device).eval()

# 이미지 전처리
transform = Compose([
    Resize(224, interpolation=InterpolationMode.BICUBIC),
    CenterCrop(224),
    lambda img: img.convert("RGB"),
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073),
              (0.26862954, 0.26130258, 0.27577711)),
])

# 텍스트 임베딩
def get_text_embedding(text):
    inputs = text_tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = text_model(**inputs)[0]
        output = output / output.norm(dim=-1, keepdim=True)
    return output.squeeze(0)

# 프레임 임베딩
def get_frame_embedding(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = vision_model(pixel_values=image_tensor)["image_embeds"]
        output = output / output.norm(dim=-1, keepdim=True)
    return output.squeeze(0)

# similarity 누적용 리스트
similarities = []

# 실시간 추론 그래프 업데이트 함수
def update_inference_plot(_): 
    global similarities, cap, text_embedding

    ret, frame = cap.read()
    if not ret:
        return

    frame_embedding = get_frame_embedding(frame)
    similarity = torch.dot(frame_embedding, text_embedding).item()
    def normalize_similarity(sim, center=0.20, sharpness=40): # 표본 기반 정규화...
        return 1 / (1 + np.exp(-sharpness * (sim - center)))
    similarity = normalize_similarity(similarity)
    similarities.append(similarity)

    # 화면에 유사도 수치 표시
    cv2.putText(frame, f"Similarity: {similarity:.3f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Webcam Inference", frame)

    ax.clear()
    ax.plot(similarities, color="crimson")
    ax.set_title("Similarity Over Time")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Cosine Similarity")
    ax.set_ylim(0, 1)

if __name__ == "__main__":
    # query = input("🔍 비교할 텍스트를 입력하세요 (예: '사람이 누워 있다'): ")
    query = "A person is facing forward with their hand open and fingers fully extended"
    text_embedding = get_text_embedding(query)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠 열기 실패")
        exit()

    # 추론 시각화 설정
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(8, 4))

    ani = FuncAnimation(fig, update_inference_plot, interval=500)  # 500ms 간격
    plt.show()

    # 종료
    cap.release()
    cv2.destroyAllWindows()