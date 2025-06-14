import cv2
import torch
import numpy as np
import psutil
import time
import csv
import pandas as pd
from collections import deque
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection

# 모델 및 장치 설정
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

# 모니터링 관련 변수
similarities = []
log_data = []  # CSV로 저장할 데이터
frame_interval_sec = 0.5
last_time = time.time()
frame_count = 0
fps_window = deque(maxlen=30)

def get_ram_usage():
    return psutil.virtual_memory().percent

def update_plot(frame_idx):
    global similarities, cap, text_embedding, last_time, frame_count

    current_time = time.time()
    if current_time - last_time < frame_interval_sec:
        return
    last_time = current_time

    ret, frame = cap.read()
    if not ret:
        return

    frame_count += 1
    fps_window.append(current_time)
    if len(fps_window) >= 2:
        fps = len(fps_window) / (fps_window[-1] - fps_window[0] + 1e-8)
    else:
        fps = 0.0

    # === 추론 시간 측정 시작 ===
    infer_start = time.time()
    frame_embedding = get_frame_embedding(frame)
    infer_latency = (time.time() - infer_start) * 1000  # ms 단위
    # ===========================

    similarity = torch.dot(frame_embedding, text_embedding).item()
    similarities.append(similarity)

    ram_usage = get_ram_usage()

    # 로그 저장용 데이터
    log_data.append({
        "timestamp": current_time,
        "similarity": similarity,
        "fps": fps,
        "latency_ms": infer_latency,
        "ram_percent": ram_usage,
    })

    # 모니터링 정보 표시
    monitor_text = f"Sim: {similarity:.3f} | FPS: {fps:.2f} | Lat: {infer_latency:.1f}ms | RAM: {ram_usage:.1f}%"
    cv2.putText(frame, monitor_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Webcam Inference", frame)

    ax.clear()
    ax.plot(similarities, color="crimson")
    ax.set_title("Similarity Over Time")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Cosine Similarity")
    ax.set_ylim(0, 1)

def save_log_to_csv(filename="monitoring_log.csv"):
    fieldnames = ["timestamp", "similarity", "fps", "latency_ms", "ram_percent"]
    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_data)
    print(f"\n📁 로그 저장 완료: {filename}")

def summarize_csv(filename="monitoring_log.csv"):
    df = pd.read_csv(filename)
    print("\n📊 [모니터링 요약 통계]")
    for column in ["fps", "latency_ms", "ram_percent"]:
        print(f"\n▶ {column}:")
        print(f"  평균: {df[column].mean():.2f}")
        print(f"  최소: {df[column].min():.2f}")
        print(f"  최대: {df[column].max():.2f}")

if __name__ == "__main__":
    query = "A person is facing forward with their hand open and fingers fully extended"
    text_embedding = get_text_embedding(query)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠 열기 실패")
        exit()

    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(8, 4))

    ani = FuncAnimation(fig, update_plot, interval=50)
    plt.show()

    cap.release()
    cv2.destroyAllWindows()

    # CSV 저장 및 통계 출력
    csv_filename = "monitoring_log.csv"
    save_log_to_csv(csv_filename)
    summarize_csv(csv_filename)