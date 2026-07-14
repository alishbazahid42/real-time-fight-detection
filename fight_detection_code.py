"""
Real-Time Fight Detection for Smart Surveillance Interfaces
CNN-LSTM Implementation — Memory-Efficient Version (Generator-based)
Dataset: RWF-2000 (primary) + UCF-Crime Fighting + UCF-Crime Assault
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog
from datetime import datetime
import threading
import random

# ─────────────────────────────────────────────
# 1.  CONFIGURATION
# ─────────────────────────────────────────────
IMG_SIZE   = 112          # reduced from 224 → saves 4x memory, still works well
FRAMES     = 16
BATCH_SIZE = 4            # small batch to avoid RAM issues
EPOCHS     = 10
LR         = 1e-4
CLASSES    = ['NonFight', 'Fight']

DATA_ROOT  = "data"
MODEL_PATH = "cnn_lstm_fight.h5"


# ─────────────────────────────────────────────
# 2.  VIDEO LOADING
# ─────────────────────────────────────────────
def load_video_frames(path, n_frames=FRAMES, img_size=IMG_SIZE):
    """Load n_frames uniformly sampled from a video.
    Returns float32 (n_frames, H, W, 3) or None on failure.
    """
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return None

    indices = (list(range(total)) + [total-1]*(n_frames-total)
               if total < n_frames
               else np.linspace(0, total-1, n_frames, dtype=int))

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        frame = cv2.resize(frame, (img_size, img_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        frame = (frame - mean) / std
        frames.append(frame)

    cap.release()
    return np.array(frames, dtype=np.float32)


# ─────────────────────────────────────────────
# 3.  DATA GENERATOR  ← key fix for memory
# ─────────────────────────────────────────────
def get_file_list(split):
    """Return list of (filepath, label) for a split."""
    items = []
    label_map = {'nonfight': 0, 'fight': 1}
    for cls, label in label_map.items():
        folder = os.path.join(DATA_ROOT, split, cls)
        if not os.path.isdir(folder):
            print(f"[WARNING] Folder not found: {folder}")
            continue
        files = [f for f in os.listdir(folder)
                 if f.lower().endswith(('.avi', '.mp4', '.mkv'))]
        print(f"  {split}/{cls}: {len(files)} files")
        for f in files:
            items.append((os.path.join(folder, f), label))
    return items


def data_generator(file_list, batch_size=BATCH_SIZE, shuffle=True):
    """Yields (X_batch, y_batch) — loads videos one batch at a time.
    Never loads the full dataset into RAM.
    """
    if shuffle:
        random.shuffle(file_list)

    batch_x, batch_y = [], []
    for path, label in file_list:
        clip = load_video_frames(path)
        if clip is None:
            continue
        batch_x.append(clip)
        batch_y.append(label)

        if len(batch_x) == batch_size:
            yield (np.array(batch_x, dtype=np.float32),
                   np.array(batch_y, dtype=np.float32))
            batch_x, batch_y = [], []

    # Leftover partial batch
    if batch_x:
        yield (np.array(batch_x, dtype=np.float32),
               np.array(batch_y, dtype=np.float32))


def make_tf_dataset(split, batch_size=BATCH_SIZE, shuffle=True):
    """Wrap generator into a tf.data.Dataset."""
    file_list = get_file_list(split)
    if not file_list:
        raise ValueError(f"No videos found for split '{split}'. "
                         f"Check your data/ folder.")

    def gen():
        yield from data_generator(file_list, batch_size=batch_size,
                                  shuffle=shuffle)

    output_sig = (
        tf.TensorSpec(shape=(None, FRAMES, IMG_SIZE, IMG_SIZE, 3),
                      dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_sig)
    return ds, len(file_list)


# ─────────────────────────────────────────────
# 4.  MODEL
# ─────────────────────────────────────────────
def build_model(frames=FRAMES, img_size=IMG_SIZE):
    base = MobileNetV2(include_top=False, weights='imagenet',
                       input_shape=(img_size, img_size, 3),
                       pooling='avg')
    for layer in base.layers[:-30]:
        layer.trainable = False

    inp = layers.Input(shape=(frames, img_size, img_size, 3),
                       name='video_input')
    x = layers.TimeDistributed(base, name='cnn_features')(inp)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(256, return_sequences=True, name='lstm_1')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(128, return_sequences=False, name='lstm_2')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='sigmoid', name='output')(x)

    model = Model(inp, out, name='CNN_LSTM_FightDetector')
    model.compile(
        optimizer=Adam(learning_rate=LR),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    return model


# ─────────────────────────────────────────────
# 5.  TRAINING
# ─────────────────────────────────────────────
def train():
    print("\nBuilding datasets (videos load batch-by-batch — no RAM crash)...")
    train_ds, n_train = make_tf_dataset('train', shuffle=True)
    val_ds,   n_val   = make_tf_dataset('val',   shuffle=False)

    steps_per_epoch  = max(1, n_train // BATCH_SIZE)
    validation_steps = max(1, n_val   // BATCH_SIZE)
    print(f"  Train samples: {n_train}  |  Val samples: {n_val}")
    print(f"  Steps/epoch: {steps_per_epoch}  |  Val steps: {validation_steps}")

    model = build_model()
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3,
                      restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, save_best_only=True,
                        monitor='val_accuracy'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=2, min_lr=1e-6)
    ]

    # Class weights
    train_files = get_file_list('train')
    n_neg = sum(1 for _, l in train_files if l == 0)
    n_pos = sum(1 for _, l in train_files if l == 1)
    total = n_neg + n_pos
    class_weight = {0: total / (2 * n_neg), 1: total / (2 * n_pos)}
    print(f"  Class weights: {class_weight}")

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=callbacks,
        class_weight=class_weight
    )

    plot_training(history)
    print(f"\nModel saved to: {MODEL_PATH}")
    return model


def plot_training(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history['accuracy'],     label='Train Acc')
    axes[0].plot(history.history['val_accuracy'], label='Val Acc')
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(history.history['loss'],     label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    plt.show()
    print("Saved: training_curves.png")


# ─────────────────────────────────────────────
# 6.  EVALUATION
# ─────────────────────────────────────────────
def evaluate(model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}. Run --mode train first.")
        return

    model = tf.keras.models.load_model(model_path)
    print("Loading test data batch by batch...")

    test_files = get_file_list('test')
    y_true, y_pred_list = [], []

    for path, label in test_files:
        clip = load_video_frames(path)
        if clip is None:
            continue
        clip_input = np.expand_dims(clip, axis=0)   # (1, T, H, W, 3)
        prob = float(model.predict(clip_input, verbose=0)[0][0])
        y_true.append(label)
        y_pred_list.append(1 if prob > 0.5 else 0)

    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred_list, target_names=CLASSES))

    cm = confusion_matrix(y_true, y_pred_list)
    print("Confusion Matrix:")
    print(cm)
    plot_confusion_matrix(cm)


def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASSES); ax.set_yticklabels(CLASSES)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()
    print("Saved: confusion_matrix.png")


# ─────────────────────────────────────────────
# 7.  REAL-TIME INFERENCE
# ─────────────────────────────────────────────
class FightDetector:
    def __init__(self, model_path=MODEL_PATH, window=FRAMES,
                 threshold=0.5):
        self.model     = tf.keras.models.load_model(model_path)
        self.window    = window
        self.threshold = threshold
        self.buffer    = []
        self._mean     = np.array([0.485, 0.456, 0.406], np.float32)
        self._std      = np.array([0.229, 0.224, 0.225], np.float32)

    def preprocess_frame(self, frame):
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        return (frame - self._mean) / self._std

    def push(self, frame):
        self.buffer.append(self.preprocess_frame(frame))
        if len(self.buffer) < self.window:
            return None
        if len(self.buffer) > self.window:
            self.buffer = self.buffer[-self.window:]
        clip = np.expand_dims(np.array(self.buffer), axis=0)
        prob = float(self.model.predict(clip, verbose=0)[0][0])
        return ('Fight' if prob >= self.threshold else 'Non-Fight'), prob


# ─────────────────────────────────────────────
# 8.  SURVEILLANCE DASHBOARD
# ─────────────────────────────────────────────
class SurveillanceDashboard:
    def __init__(self, root, detector):
        self.root      = root
        self.detector  = detector
        self.cap       = None
        self.running   = False
        self.log_items = []
        root.title("Smart Surveillance – Fight & Assault Detection")
        root.configure(bg="#1a1a2e")
        self._build_ui()

    def _build_ui(self):
        tk.Label(self.root,
                 text="SMART SURVEILLANCE SYSTEM — FIGHT & ASSAULT DETECTION",
                 font=("Helvetica", 13, "bold"),
                 fg="#00d4ff", bg="#1a1a2e").grid(
                 row=0, column=0, columnspan=2, pady=8)

        self.canvas = tk.Canvas(self.root, width=640, height=480,
                                bg="#000000", bd=2, relief="ridge")
        self.canvas.grid(row=1, column=0, padx=10, pady=5)

        right = tk.Frame(self.root, bg="#1a1a2e")
        right.grid(row=1, column=1, padx=10, pady=5, sticky="ns")

        tk.Label(right, text="Detection Confidence",
                 font=("Helvetica", 10, "bold"),
                 fg="#ffffff", bg="#1a1a2e").pack(pady=(0, 4))
        self.conf_var = tk.DoubleVar(value=0.0)
        ttk.Progressbar(right, orient="horizontal", length=200,
                        mode="determinate", variable=self.conf_var,
                        maximum=1.0).pack()
        self.conf_label = tk.Label(right, text="0.00%",
                                   font=("Helvetica", 10),
                                   fg="#aaaaaa", bg="#1a1a2e")
        self.conf_label.pack(pady=(2, 10))

        self.status_var = tk.StringVar(value="● No Activity Detected")
        self.status_lbl = tk.Label(right, textvariable=self.status_var,
                                   font=("Helvetica", 11, "bold"),
                                   fg="#00ff88", bg="#1a1a2e", width=24)
        self.status_lbl.pack(pady=8)

        tk.Label(right, text="Incident Log",
                 font=("Helvetica", 10, "bold"),
                 fg="#ffffff", bg="#1a1a2e").pack(pady=(10, 4))
        self.log_box = tk.Listbox(right, width=28, height=12,
                                  bg="#0f0f1a", fg="#ffffff",
                                  font=("Courier", 8),
                                  selectbackground="#003366")
        self.log_box.pack()

        ctrl = tk.Frame(self.root, bg="#1a1a2e")
        ctrl.grid(row=2, column=0, columnspan=2, pady=8)
        for txt, cmd, col in [
            ("Open Video File", self.open_file,  "#0066cc"),
            ("Use Webcam",      self.use_webcam, "#006633"),
            ("Stop",            self.stop,       "#cc0000"),
        ]:
            tk.Button(ctrl, text=txt, command=cmd, bg=col,
                      fg="white", font=("Helvetica", 10)).pack(
                      side="left", padx=6)

    def open_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov")])
        if path:
            self.start_stream(path)

    def use_webcam(self):
        self.start_stream(0)

    def start_stream(self, source):
        self.stop()
        self.cap     = cv2.VideoCapture(source)
        self.running = True
        threading.Thread(target=self._stream_loop, daemon=True).start()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def _stream_loop(self):
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            result     = self.detector.push(frame)
            is_fight   = False
            confidence = 0.0
            if result:
                label, confidence = result
                is_fight = (label == 'Fight')

            display = frame.copy()
            if is_fight:
                h, w = display.shape[:2]
                cv2.rectangle(display, (10, 10), (w-10, h-10),
                              (0, 0, 255), 4)
                cv2.putText(display,
                            f"FIGHT/ASSAULT  {confidence*100:.0f}%",
                            (20, 45), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 0, 255), 2)

            from PIL import Image, ImageTk
            rgb = cv2.cvtColor(cv2.resize(display, (640, 480)),
                               cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.root.after(0, self._update_ui, img, is_fight, confidence)
        self.running = False

    def _update_ui(self, img, is_fight, confidence):
        self.canvas.create_image(0, 0, anchor="nw", image=img)
        self.canvas.image = img
        self.conf_var.set(confidence)
        self.conf_label.config(text=f"{confidence*100:.1f}%")
        if is_fight:
            self.status_var.set("⚠ FIGHT / ASSAULT DETECTED!")
            self.status_lbl.config(fg="#ff4444")
            ts    = datetime.now().strftime("%H:%M:%S")
            entry = f"{ts}  Cam-1  {confidence*100:.0f}%"
            self.log_items.insert(0, entry)
            self.log_box.insert(0, entry)
            self.log_box.itemconfig(0, fg="#ff4444")
        else:
            self.status_var.set("● No Activity Detected")
            self.status_lbl.config(fg="#00ff88")


# ─────────────────────────────────────────────
# 9.  MAIN
# ─────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fight & Assault Detection")
    parser.add_argument('--mode',
                        choices=['train', 'eval', 'gui', 'demo'],
                        default='demo')
    args = parser.parse_args()

    if args.mode == 'train':
        train()

    elif args.mode == 'eval':
        evaluate()

    elif args.mode == 'gui':
        if not os.path.exists(MODEL_PATH):
            print(f"[ERROR] Model not found. Run --mode train first.")
            return
        detector = FightDetector()
        root = tk.Tk()
        SurveillanceDashboard(root, detector)
        root.mainloop()

    elif args.mode == 'demo':
        print("=== DEMO MODE (synthetic data) ===")
        model = build_model()
        model.summary()
        X = np.random.rand(2, FRAMES, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
        preds = model.predict(X, verbose=0)
        print(f"Input shape : {X.shape}")
        print(f"Predictions : {preds.flatten().tolist()}")
        print("Pipeline OK. Run --mode train to train on real data.")


if __name__ == '__main__':
    main()
