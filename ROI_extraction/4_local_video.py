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
import argparse
import os

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

background_frame = None
change_thresh = 30
persistence_map = None
roi_threshold = 15
cap = None
ax = None
out = None
args = None


def update_plot(_):
    global similarities, cap, text_embedding, last_time, frame_count
    global background_frame, persistence_map, out

    current_time = time.time()
    if current_time - last_time < frame_interval_sec:
        return
    last_time = current_time

    ret, frame = cap.read()
    if not ret:
        print("🎞️ 영상 종료")
        plt.close()
        return

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if background_frame is None:
        background_frame = frame_gray.copy()
        persistence_map = np.zeros_like(frame_gray, dtype=np.uint8)
        print("✅ 배경 프레임 초기화 완료")
        cv2.imshow("Initialized Background", background_frame)
        cv2.waitKey(1)
        return

    diff = cv2.absdiff(frame_gray, background_frame)
    motion_mask = (diff > change_thresh).astype(np.uint8)
    persistence_map[motion_mask == 1] += 1
    persistence_map[motion_mask == 0] = 0
    roi_mask = (persistence_map >= roi_threshold).astype(np.uint8)

    if np.count_nonzero(roi_mask) == 0:
        return

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(roi_mask, connectivity=8)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area >= 500:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    frame_count += 1
    fps_window.append(current_time)
    fps = len(fps_window) / (fps_window[-1] - fps_window[0] + 1e-8) if len(fps_window) >= 2 else 0.0

    infer_start = time.time()
    # frame_embedding = get_frame_embedding(frame)
    # infer_latency = (time.time() - infer_start) * 1000
    # similarity = torch.dot(frame_embedding, text_embedding).item()
    roi_similarities = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area >= 500:
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0:
                continue
            try:
                emb = get_frame_embedding(roi)
                sim = torch.dot(emb, text_embedding).item()
                roi_similarities.append(sim)
            except Exception as e:
                print(f"⚠️ ROI 추론 오류: {e}")
    infer_latency = (time.time() - infer_start) * 1000  # ⬅️ ROI 추론 종료 시점
    # 최종 similarity는 ROI 중 최고값
    similarity = max(roi_similarities) if roi_similarities else 0.0

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
    if out is not None:
        out.write(frame)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", type=str, help="Local video file path")
    args = parser.parse_args()

    # query = "A person is facing forward with their hand open and fingers fully extended"
    query = "A person is laying down"
    text_embedding = get_text_embedding(query)

    if args.local:
        video_path = args.local
        if not os.path.exists(video_path):
            print(f"❌ 파일을 찾을 수 없음: {video_path}")
            exit()
        cap = cv2.VideoCapture(video_path)
        out_path = os.path.join("./store", os.path.basename(video_path))
        os.makedirs("./store", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(out_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    else:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("웹캠/비디오 열기 실패")
        exit()

    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(8, 4))

    ani = FuncAnimation(fig, update_plot, interval=50)
    plt.show()

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

    csv_filename = "./monitoring_log.csv"
    save_log_to_csv(csv_filename)
    summarize_csv(csv_filename)
