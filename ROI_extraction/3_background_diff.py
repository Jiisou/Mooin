# 변경사항 요약:
# - Background 초기화 프레임 추가
# - 변화 감지 기반 trigger 조건 추가
# - 변화 지속성 기반 Dynamic ROI 추적 추가
# - ROI가 활성화되었을 때만 추론 수행

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

# ================================
# 초기 설정
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text_tokenizer = CLIPTokenizer.from_pretrained("Searchium-ai/clip4clip-webvid150k")
text_model = CLIPTextModelWithProjection.from_pretrained("Searchium-ai/clip4clip-webvid150k").to(device).eval()
vision_model = CLIPVisionModelWithProjection.from_pretrained("Searchium-ai/clip4clip-webvid150k").to(device).eval()

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

def get_ram_usage():
    return psutil.virtual_memory().percent

# ================================
# 전역 변수 및 초기화
# ================================
similarities = []
log_data = []
frame_interval_sec = 0.5
last_time = time.time()
frame_count = 0
fps_window = deque(maxlen=30)

# 배경 설정 관련
background_frame = None
change_thresh = 100  # 픽셀 단위 차이 임계값
persistence_map = None
roi_threshold = 15  # ROI로 간주되기 위한 연속 변화 프레임 수

def update_plot(_):
    global similarities, cap, text_embedding, last_time, frame_count, background_frame, persistence_map

    current_time = time.time()
    if current_time - last_time < frame_interval_sec:
        return
    last_time = current_time

    ret, frame = cap.read()
    if not ret:
        return

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if background_frame is None:
        background_frame = frame_gray.copy()
        persistence_map = np.zeros_like(frame_gray, dtype=np.uint8)
        print("✅ 배경 프레임 초기화 완료")

        cv2.imshow("Initialized Background", background_frame)
        cv2.waitKey(1)  # 한 프레임만 열고 닫지 않게 유지
        return

    # 변화 감지
    diff = cv2.absdiff(frame_gray, background_frame)
    motion_mask = (diff > change_thresh).astype(np.uint8)

    # 지속성 추적
    persistence_map[motion_mask == 1] += 1
    persistence_map[motion_mask == 0] = 0
    roi_mask = (persistence_map >= roi_threshold).astype(np.uint8)

    if np.count_nonzero(roi_mask) == 0:
        return  # 변화 지속 ROI가 없으면 skip
    
    # === ROI Bounding Box 시각화 ===
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(roi_mask, connectivity=8)
    for i in range(1, num_labels):  # label 0은 background
        x, y, w, h, area = stats[i]
        if area >= 500:  # 너무 작은 영역은 무시
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # FPS 계산
    frame_count += 1
    fps_window.append(current_time)
    fps = len(fps_window) / (fps_window[-1] - fps_window[0] + 1e-8) if len(fps_window) >= 2 else 0.0

    # ROI 기반 클립 추론
    infer_start = time.time()
    frame_embedding = get_frame_embedding(frame)
    infer_latency = (time.time() - infer_start) * 1000
    similarity = torch.dot(frame_embedding, text_embedding).item()

    similarities.append(similarity)
    ram_usage = get_ram_usage()

    log_data.append({
        "timestamp": current_time,
        "similarity": similarity,
        "fps": fps,
        "latency_ms": infer_latency,
        "ram_percent": ram_usage,
    })

    monitor_text = f"Sim: {similarity:.3f} | FPS: {fps:.2f} | Lat: {infer_latency:.1f}ms | RAM: {ram_usage:.1f}%"
    cv2.putText(frame, monitor_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Dynamic ROI Inference", frame)

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

    csv_filename = "./monitoring_log.csv"
    save_log_to_csv(csv_filename)
    summarize_csv(csv_filename)