
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import cv2
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import uvicorn
import io
import time
import base64
from PIL import Image
from spellchecker import SpellChecker

app = FastAPI(title="Neural Scribe - HTR API")
spell = SpellChecker()

IMG_H = 32
IMG_W = 128
TIME_STEPS = 64

CHARSET = [' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
num_to_char = {i: c for i, c in enumerate(CHARSET)}
BLANK = len(CHARSET)

try:
    prediction_model = tf.keras.models.load_model("prediksi.h5", compile=False)
    print("Model loaded successfully")
    
    # Create feature extractor for intermediate layers
    # We need: Conv2D_1 (spatial), BiLSTM_1 (temporal)
    # Let's dynamically find the layer names for safety.
    layer_names = [layer.name for layer in prediction_model.layers]
    conv_layer = [name for name in layer_names if 'conv2d' in name][0] # first conv
    lstm_layer = [name for name in layer_names if 'bidirectional' in name][-1] # last bilstm
    
    extractor = tf.keras.models.Model(
        inputs=prediction_model.inputs,
        outputs=[
            prediction_model.get_layer(conv_layer).output,
            prediction_model.get_layer(lstm_layer).output,
            prediction_model.output
        ]
    )
except Exception as e:
    print("Failed to load model:", e)
    prediction_model = None
    extractor = None

def simple_levenshtein(s1, s2):
    if len(s1) < len(s2): return simple_levenshtein(s2, s1)
    if len(s2) == 0: return len(s1)
    prev = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            ins = prev[j + 1] + 1
            dele = curr[j] + 1
            subs = prev[j] + (c1 != c2)
            curr.append(min(ins, dele, subs))
        prev = curr
    return prev[-1]

def preprocess_and_get_math(pil_img):
    img = np.array(pil_img.convert("L"))
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Calculate Otsu manually for telemetry math
    hist = cv2.calcHist([blurred], [0], None, [256], [0, 256])
    hist_norm = hist.ravel()/hist.sum()
    Q = hist_norm.cumsum()
    bins = np.arange(256)
    fn_min = np.inf
    thresh = -1
    for i in range(1, 256):
        p1, p2 = np.hsplit(hist_norm, [i])
        q1, q2 = Q[i], Q[255]-Q[i]
        if q1 == 0 or q2 == 0: continue
        b1, b2 = np.hsplit(bins, [i])
        m1, m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
        v1, v2 = np.sum(((b1-m1)**2)*p1)/q1, np.sum(((b2-m2)**2)*p2)/q2
        fn = v1*q1 + v2*q2
        if fn < fn_min:
            fn_min = fn
            thresh = i
            
    if fn_min == np.inf or np.isnan(fn_min):
        fn_min = 0.0
            
    # Matrix slice (5x5) before and after
    mid_y, mid_x = blurred.shape[0]//2, blurred.shape[1]//2
    mat_before = blurred[mid_y:mid_y+5, mid_x:mid_x+5].tolist()
    
    _, img_thresh = cv2.threshold(blurred, thresh, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    mat_after = (img_thresh[mid_y:mid_y+5, mid_x:mid_x+5] // 255).tolist() # 0 and 1
    
    img = img_thresh
    coords = cv2.findNonZero(255 - img)
    if coords is not None:
        x, y, w_box, h_box = cv2.boundingRect(coords)
        margin = 5
        y1, y2 = max(0, y - margin), min(img.shape[0], y + h_box + margin)
        x1, x2 = max(0, x - margin), min(img.shape[1], x + w_box + margin)
        img = img[y1:y2, x1:x2]
        
    h, w = img.shape
    scale = IMG_H / h
    new_w = int(round(w * scale))
    new_w = max(1, min(IMG_W, new_w))
    resized = cv2.resize(img, (new_w, IMG_H), interpolation=cv2.INTER_AREA)
    canvas = np.ones((IMG_H, IMG_W), dtype=np.uint8) * 255
    canvas[:, :new_w] = resized
    final_img = canvas.astype(np.float32) / 255.0
    
    math_data = {
        "otsu_thresh": int(thresh),
        "variance_min": float(fn_min),
        "mat_before": mat_before,
        "mat_after": mat_after
    }
    return final_img, canvas, math_data

def decode_ctc_math(probs):
    # Returns raw string, decoded text, and softmax sample
    input_len = np.ones(probs.shape[0]) * probs.shape[1]
    decoded, _ = tf.keras.backend.ctc_decode(probs, input_length=input_len, greedy=True)
    decoded_arr = decoded[0].numpy()[0]
    
    # Raw sequence tracking for CTC math (showing blank labels etc)
    raw_seq = np.argmax(probs[0], axis=-1)
    raw_chars = []
    for idx in raw_seq:
        if idx == BLANK: raw_chars.append('-')
        else: raw_chars.append(num_to_char.get(idx, '?'))
    
    # Collapse string
    text = ""
    for idx in decoded_arr:
        if idx < 0 or idx == BLANK: continue
        text += num_to_char.get(idx, "")
        
    # Get softmax vector sample for a time step
    softmax_sample = np.round(probs[0][10][:5], 4).tolist() # time step 10, first 5 classes
        
    return text, "".join(raw_chars), softmax_sample

@app.post("/api/predict")
async def predict_image(file: UploadFile = File(...)):
    start_time = time.time()
    contents = await file.read()
    image_pil = Image.open(io.BytesIO(contents))
    
    # Preprocessing & Vision Math
    img_norm, preview_img, vision_math = preprocess_and_get_math(image_pil)
    img_input = np.expand_dims(img_norm, axis=(0, -1))
    
    _, buffer = cv2.imencode('.png', preview_img)
    base64_img = base64.b64encode(buffer).decode('utf-8')
    
    # Inference & Deep Math extraction
    cnn_sample = []
    lstm_sample = []
    text = ""
    raw_path = ""
    softmax = []
    
    if extractor:
        conv_out, lstm_out, probs = extractor(img_input, training=False)
        # Spatial Math (Conv2D): Get a 3x3 filter activation map from channel 0
        cnn_map = conv_out[0, :3, :3, 0].numpy()
        cnn_sample = np.round(cnn_map, 3).tolist()
        
        # Temporal Math (BiLSTM): Get first 5 values of hidden state vector at t=32
        lstm_vec = lstm_out[0, 32, :5].numpy()
        lstm_sample = np.round(lstm_vec, 3).tolist()
        
        # CTC Math
        text, raw_path, softmax = decode_ctc_math(probs.numpy())
    
    corrected_pred = spell.correction(text)
    if not corrected_pred:
        corrected_pred = text
        
    # Lexical Decoding Math (Levenshtein matrix)
    # Build a mini DP table for the first 5 chars
    raw_short = text[:5] if text else "none"
    tgt_short = corrected_pred[:5] if corrected_pred else "none"
    dp_table = []
    if len(raw_short) > 0:
        prev = list(range(len(tgt_short) + 1))
        dp_table.append(prev)
        for i, c1 in enumerate(raw_short):
            curr = [i + 1]
            for j, c2 in enumerate(tgt_short):
                curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
            dp_table.append(curr)
            prev = curr

    inference_time = round((time.time() - start_time) * 1000)
    
    telemetry = {
        "vision": {
            "thresh": vision_math["otsu_thresh"],
            "variance_min": vision_math["variance_min"],
            "mat_before": vision_math["mat_before"],
            "mat_after": vision_math["mat_after"]
        },
        "spatial": {
            "tensor_map": cnn_sample
        },
        "temporal": {
            "hidden_state": lstm_sample
        },
        "ctc": {
            "raw_path": raw_path,
            "softmax_sample": softmax
        },
        "nlp": {
            "distance": simple_levenshtein(text, corrected_pred) if text else 0,
            "dp_grid": dp_table,
            "final_text": corrected_pred,
            "raw_word": raw_short,
            "target": tgt_short
        }
    }
    
    return {
        "prediction": corrected_pred, 
        "raw": text,
        "processed_image": f"data:image/png;base64,{base64_img}",
        "latency_ms": inference_time,
        "telemetry": telemetry
    }

@app.get("/")
async def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
