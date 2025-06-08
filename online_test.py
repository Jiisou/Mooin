import cv2
import torch
import numpy as np
from PIL import Image
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
import time

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 불러오기
text_tokenizer = CLIPTokenizer.from_pretrained("Searchium-ai/clip4clip-webvid150k")
text_model = CLIPTextModelWithProjection.from_pretrained("Searchium-ai/clip4clip-webvid150k").to(device).eval()
vision_model = CLIPVisionModelWithProjection.from_pretrained("Searchium-ai/clip4clip-webvid150k").to(device).eval()

# 전처리 함수
transform = Compose([
    Resize(224, interpolation=InterpolationMode.BICUBIC),
    CenterCrop(224),
    lambda img: img.convert("RGB"),
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073),
              (0.26862954, 0.26130258, 0.27577711)),
])

def get_text_embedding(text):
    inputs = text_tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = text_model(**inputs)[0]
        output = output / output.norm(dim=-1, keepdim=True)
    return output.squeeze(0)

def get_frame_embedding(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = vision_model(pixel_values=image_tensor)["image_embeds"]
        output = output / output.norm(dim=-1, keepdim=True)
    return output.squeeze(0)

# Real-time inference loop
def run_online_inference(text_prompt):
    text_embedding = get_text_embedding(text_prompt)

    cap = cv2.VideoCapture(0)  # 기본 카메라
    if not cap.isOpened():
        print("웹캠 열기에 실패했습니다.")
        return

    print("🔍 실시간 추론을 시작합니다. 'q'를 누르면 종료됩니다.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        # 추론
        frame_embedding = get_frame_embedding(frame)
        similarity = torch.dot(frame_embedding, text_embedding).item()

        # 화면에 유사도 표시
        cv2.putText(frame, f"Similarity: {similarity:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("VLM Real-Time Inference", frame)

        # 종료 조건
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 사용 예시
if __name__ == "__main__":
    query = input("🔎 비교할 텍스트를 입력하세요 (예: '사람이 누워 있다'): ")
    run_online_inference(query)
