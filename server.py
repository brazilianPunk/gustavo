import os
import time
import logging
import threading
import cv2
import numpy as np
import requests

from flask import Flask, render_template, Response, request, redirect, url_for
from supabase import create_client
from deepface import DeepFace

face_model = DeepFace.build_model("Facenet")

latest_frame = None
display_text = None
display_color = (0, 255, 0)
stop_recognition = False
# -----------------------
# Configuração
# -----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SUPABASE_URL = "https://udpbhizhyaesheidalho.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...."  # substitua pela sua chave
SUPABASE_BUCKET = "Rostos"
REGISTERED_PATH = "capture (3).jpg"
REGISTERED_NAME = "Gustavo"

ESP32_CAM_URL = "http://192.168.1.11/stream"
INTERVALO_SEGUNDOS = 2.0
SIMILARITY_THRESHOLD = 0.65

app = Flask(__name__)
supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

lock = threading.Lock()
reconhecimento_manual = False
registered_embedding = None
registered_name = None
last_result = None  # (similaridade, reconhecido)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# -----------------------
# Funções auxiliares
# -----------------------
def get_public_url(bucket, path):
    try:
        res = supabase.storage.from_(bucket).get_public_url(path)
        if isinstance(res, dict):
            return res.get("publicUrl") or res.get("public_url")
        return res
    except Exception as e:
        logging.error("Erro gerando public url: %s", e)
        return None

# Corrige frames sem tabela Huffman (DHT)
def ensure_dht(jpeg_bytes):
    DHT = bytes.fromhex(
        "FFC4 01A2 00000105 01010101 01010101 01010101 01010101 01010101"
        "01010101 01010101 01010101 01010101 01010101 01010101 01010101"
        "01010101 01010101 01010101 01010101 01010101 01010101 01010101"
        "01010101 01010101 01010101 01010101 01010101 01010101 01010101"
        "01010101 01010101 01010101 01010101 01010101 01010101 01010101"
        "01010101 01010101 01010101 01010101 01010101 01010101 01010101"
        "01010101 01010101 01010101 01010101 01010101 01010101 01010101"
        "01010101 01010101 01010101 01010101 01010101 01010101 01010101"
        "01010101 01010101 01010101 01010101 01010101 01010101 01010101"
    )
    # insere DHT antes do marcador SOS (FFDA)
    sos_index = jpeg_bytes.find(b"\xFF\xDA")
    if sos_index != -1:
        jpeg_bytes = jpeg_bytes[:sos_index] + DHT + jpeg_bytes[sos_index:]
    return jpeg_bytes

def download_image_bgr(url):
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    arr = np.asarray(bytearray(r.content), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def load_registered_embedding():
    global registered_embedding, registered_name
    try:
        public_url = get_public_url(SUPABASE_BUCKET, REGISTERED_PATH)
        if not public_url:
            logging.error("Não foi possível obter URL pública para %s/%s", SUPABASE_BUCKET, REGISTERED_PATH)
            return False
        img = download_image_bgr(public_url)
        rep = DeepFace.represent(img_path=img, model_name="Facenet", enforce_detection=False)[0]
        registered_embedding = np.array(rep["embedding"], dtype=np.float32)
        registered_name = REGISTERED_NAME
        logging.info("✅ Embedding carregado para: %s", registered_name)
        return True
    except Exception as e:
        logging.error("Erro ao carregar rosto cadastrado: %s", e)
        return False

load_registered_embedding()

# -----------------------
# Thread para análise facial
# -----------------------
def analisar_rosto_async(frame, faces):
    global registered_embedding, registered_name, last_result
    try:
        if registered_embedding is None:
            return
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            if face_img.size == 0:
                continue
            rep = DeepFace.represent(img_path=face_img, model_name="Facenet", enforce_detection=False)[0]
            emb = np.array(rep["embedding"], dtype=np.float32)
            sim = np.dot(registered_embedding, emb) / (np.linalg.norm(registered_embedding) * np.linalg.norm(emb) + 1e-10)
            with lock:
                last_result = (sim, sim >= SIMILARITY_THRESHOLD)
    except Exception as e:
        logging.error("Erro DeepFace async: %s", e)

# -----------------------
# Gera o vídeo para o navegador
# -----------------------
def reconhecimento_thread_func():
    global latest_frame, display_text, display_color
    while not stop_recognition:
        if latest_frame is None:
            time.sleep(0.1)
            continue

        frame = latest_frame.copy()

        try:
            # detecção de rosto rápida
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face_img = frame[y:y+h, x:x+w]

                rep = DeepFace.represent(img_path=face_img, model_name="Facenet",
                                         enforce_detection=False, model=face_model)[0]
                emb = np.array(rep["embedding"], dtype=np.float32)

                sim = np.dot(registered_embedding, emb) / (
                    np.linalg.norm(registered_embedding) * np.linalg.norm(emb) + 1e-10
                )

                if sim >= SIMILARITY_THRESHOLD:
                    display_text = f"Acesso Permitido ({registered_name}) {sim:.2f}"
                    display_color = (0, 255, 0)
                else:
                    display_text = f"Desconhecido ({sim:.2f})"
                    display_color = (0, 0, 255)
            else:
                display_text = None
        except Exception as e:
            print("[DeepFace thread] Erro:", e)
        time.sleep(INTERVALO_SEGUNDOS)


def processar_video():
    global latest_frame, display_text, display_color, stop_recognition

    cap = cv2.VideoCapture(ESP32_CAM_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Erro ao conectar ESP32-CAM.")
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", blank)
        while True:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            time.sleep(0.1)

    # inicia thread do reconhecimento
    if reconhecimento_manual:
        stop_recognition = False
        threading.Thread(target=reconhecimento_thread_func, daemon=True).start()
    else:
        stop_recognition = True

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        latest_frame = frame.copy()

        # adiciona overlay
        if display_text:
            cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, display_color, 2)

        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

        time.sleep(0.05)

    stop_recognition = True
    cap.release()
# -----------------------
# Rotas Flask
# -----------------------
@app.route('/')
def index():
    with lock:
        status = {
            "reconhecimento_manual": reconhecimento_manual,
            "registered_name": registered_name,
            "similarity_threshold": SIMILARITY_THRESHOLD
        }
    return render_template('index.html', status=status)

@app.route('/video_feed')
def video_feed():
   return Response(processar_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_reconhecimento', methods=['POST'])
def toggle_reconhecimento():
    global reconhecimento_manual
    with lock:
        reconhecimento_manual = not reconhecimento_manual
        logging.info(f"Reconhecimento manual: {reconhecimento_manual}")
    return redirect(url_for('index'))


# nova rota — fora da função acima!
@app.route("/upload_frame", methods=["POST"])
def upload_frame():
    try:
        # ler bytes da imagem enviada
        file_bytes = np.frombuffer(request.data, np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if frame is None:
            logging.error("❌ Frame inválido recebido.")
            return jsonify({"status": "erro", "mensagem": "frame inválido"}), 400

        # processar reconhecimento facial
        logging.info("📸 Frame recebido da ESP32-CAM.")
        rep = DeepFace.represent(img_path=frame, model_name="Facenet", enforce_detection=False)[0]
        emb = np.array(rep["embedding"], dtype=np.float32)

        sim = np.dot(registered_embedding, emb) / (
            np.linalg.norm(registered_embedding) * np.linalg.norm(emb) + 1e-10
        )

        if sim >= SIMILARITY_THRESHOLD:
            msg = f"Acesso Permitido ({registered_name}) {sim:.2f}"
            logging.info("✅ " + msg)
        else:
            msg = f"Desconhecido ({sim:.2f})"
            logging.warning("🚨 " + msg)

        return jsonify({"status": "ok", "mensagem": msg}), 200

    except Exception as e:
        logging.error(f"Erro no upload_frame: {e}")
        return jsonify({"status": "erro", "mensagem": str(e)}), 500

# -----------------------
# Inicialização
# -----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
