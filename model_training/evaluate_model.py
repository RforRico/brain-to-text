import os
import warnings
import time
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
import editdistance

from rnn_model import GRUDecoder
from evaluate_model_helpers import *
from pyctcdecode import build_ctcdecoder

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------
# Command line arguments
# ---------------------------
parser = argparse.ArgumentParser(
    description="Evaluate RNN model with (optional) KenLM decoding via pyctcdecode."
)
parser.add_argument(
    "--model_path", type=str, default="../data/t15_pretrained_rnn_baseline"
)
parser.add_argument("--data_dir", type=str, default="../data/hdf5_data_final")
parser.add_argument("--eval_type", type=str, default="test", choices=["val", "test"])
parser.add_argument(
    "--csv_path", type=str, default="../data/t15_copyTaskData_description.csv"
)
parser.add_argument(
    "--kenlm_path",
    type=str,
    default="../data/5gram.arpa",
    help="Path to .arpa (or a folder containing .arpa). If missing, use decoder without LM.",
)
parser.add_argument("--gpu_number", type=int, default=-1)
# decoding hyper-params
parser.add_argument(
    "--alpha", type=float, default=0.7, help="LM weight (higher → stronger LM)."
)
parser.add_argument(
    "--beta",
    type=float,
    default=0.5,
    help="Token insertion bonus (higher → longer outputs).",
)
parser.add_argument(
    "--beam_width", type=int, default=20, help="Beam width used in decoding."
)
# CTC settings
parser.add_argument(
    "--blank_idx",
    type=int,
    default=0,
    help="Index of the CTC blank token in class logits.",
)
args = parser.parse_args()


# ---------------------------
# Utilities
# ---------------------------
def numpy_log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable log-softmax in numpy."""
    x_max = np.max(x, axis=axis, keepdims=True)
    y = x - x_max
    logsum = np.log(np.exp(y).sum(axis=axis, keepdims=True) + 1e-9)
    return y - logsum


def resolve_arpa_path(kenlm_path: str | None) -> str | None:
    """Return a usable .arpa path or None (fallback to no-LM). Accepts file or directory."""
    if not kenlm_path:
        return None
    p = Path(kenlm_path).expanduser()
    if p.exists():
        if p.is_file():
            return str(p.resolve())
        if p.is_dir():
            cand = list(p.glob("*.arpa")) + list(p.glob("*.arpa.gz"))
            if not cand:
                return None
            cand.sort(key=lambda q: (q.stat().st_mtime, q.stat().st_size), reverse=True)
            return str(cand[0].resolve())
    p2 = (Path.cwd() / kenlm_path).resolve()
    if p2.exists():
        if p2.is_file():
            return str(p2)
        if p2.is_dir():
            cand = list(p2.glob("*.arpa")) + list(p2.glob("*.arpa.gz"))
            if not cand:
                return None
            cand.sort(key=lambda q: (q.stat().st_mtime, q.stat().st_size), reverse=True)
            return str(cand[0])
    return None


# ---------------------------
# Load model configuration
# ---------------------------
b2txt_csv_df = pd.read_csv(args.csv_path)
model_args = OmegaConf.load(os.path.join(args.model_path, "checkpoint/args.yaml"))

# ---------------------------
# Device setup
# ---------------------------
if (
    torch.cuda.is_available()
    and args.gpu_number >= 0
    and args.gpu_number < torch.cuda.device_count()
):
    device = torch.device(f"cuda:{args.gpu_number}")
    print(f"[INFO] Using {device}")
else:
    device = torch.device("cpu")
    if args.gpu_number >= 0 and torch.cuda.is_available():
        print(f"[WARN] Requested cuda:{args.gpu_number} but not available. Using CPU.")
    else:
        print("[INFO] Using CPU")

# ---------------------------
# Define & load model
# ---------------------------
model = GRUDecoder(
    neural_dim=model_args["model"]["n_input_features"],
    n_units=model_args["model"]["n_units"],
    n_days=len(model_args["dataset"]["sessions"]),
    n_classes=model_args["dataset"]["n_classes"],
    rnn_dropout=model_args["model"]["rnn_dropout"],
    input_dropout=model_args["model"]["input_network"]["input_layer_dropout"],
    n_layers=model_args["model"]["n_layers"],
    patch_size=model_args["model"]["patch_size"],
    patch_stride=model_args["model"]["patch_stride"],
)

# PyTorch 2.6+: weights_only default is True; use trusted full load
checkpoint = torch.load(
    os.path.join(args.model_path, "checkpoint/best_checkpoint"),
    map_location=device,
    weights_only=False,
)

# Remove DataParallel / torch.compile prefixes if present
for key in list(checkpoint["model_state_dict"].keys()):
    checkpoint["model_state_dict"][key.replace("module.", "")] = checkpoint[
        "model_state_dict"
    ].pop(key)
    checkpoint["model_state_dict"][key.replace("_orig_mod.", "")] = checkpoint[
        "model_state_dict"
    ].pop(key)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# ---------------------------
# Load dataset (read once)
# ---------------------------
test_data: dict[str, dict] = {}
total_trials = 0
for session in model_args["dataset"]["sessions"]:
    session_dir = os.path.join(args.data_dir, session)
    files = [f for f in os.listdir(session_dir) if f.endswith(".hdf5")]
    if f"data_{args.eval_type}.hdf5" in files:
        eval_file = os.path.join(session_dir, f"data_{args.eval_type}.hdf5")
        data = load_h5py_file(eval_file, b2txt_csv_df)
        test_data[session] = data
        total_trials += len(data["neural_features"])
        print(
            f"Loaded {len(data['neural_features'])} {args.eval_type} trials for session {session}."
        )
print(f"Total number of {args.eval_type} trials: {total_trials}\n")

if total_trials == 0:
    raise RuntimeError(
        "No trials found. Check --data_dir, --eval_type and sessions list in args.yaml."
    )

# ---------------------------
# Forward pass: get logits (numpy float32, no grad)  — store once
# ---------------------------
with torch.no_grad():
    with tqdm(total=total_trials, desc="Predicting logits", unit="trial") as pbar:
        for session, data in test_data.items():
            data["logits_np"] = []  # list of (T, C)
            input_layer = model_args["dataset"]["sessions"].index(session)
            for trial in range(len(data["neural_features"])):
                arr = np.expand_dims(
                    data["neural_features"][trial], axis=0
                )  # (1, T, F)
                neural_input = torch.as_tensor(arr, device=device, dtype=torch.float32)
                logits = runSingleDecodingStep(
                    neural_input, input_layer, model, model_args, device
                )  # (1, T, C)
                data["logits_np"].append(
                    np.asarray(logits[0], dtype=np.float32)
                )  # (T, C)
                pbar.update(1)

# ---------------------------
# Build labels aligned to C; move blank to index 0 by reordering
# ---------------------------
first_session = next(iter(test_data.keys()))
T0, C = test_data[first_session]["logits_np"][0].shape

labels_raw = [
    LOGIT_TO_PHONEME[i] if i < len(LOGIT_TO_PHONEME) else f"PH_{i}" for i in range(C)
]

# 重排順序：把 blank_idx 移到第 0 位
blank_idx = int(args.blank_idx)
order = [blank_idx] + [i for i in range(C) if i != blank_idx]

# 依照 order 產生 labels；index 0 設為空字串（pyctcdecode 的 blank）
labels = []
for j, i in enumerate(order):
    if j == 0:
        labels.append("")
    else:
        tok = str(labels_raw[i]).strip().replace(" ", "_")
        labels.append(tok)

if len(labels) != C:
    raise RuntimeError(
        f"[ERR] labels length {len(labels)} != C={C}. Check LOGIT_TO_PHONEME / n_classes."
    )

# ---------------------------
# Build decoder (KenLM if available, otherwise no-LM) — no ctc_token_idx
# ---------------------------
arpa_path = resolve_arpa_path(args.kenlm_path)
if arpa_path is not None:
    print(f"[INFO] Loading KenLM 5-gram model from: {arpa_path}")
else:
    print(
        f"[WARN] No .arpa found (looked up from '{args.kenlm_path}'). Using decoder WITHOUT LM."
    )

try:
    decoder = build_ctcdecoder(labels=labels, kenlm_model_path=arpa_path)
    try:
        decoder.set_lm_alpha_beta(alpha=args.alpha, beta=args.beta)
    except Exception:
        pass
except Exception as e:
    print("[WARN] build_ctcdecoder failed, fallback to NO LM:", repr(e))
    decoder = build_ctcdecoder(labels=labels, kenlm_model_path=None)

# ---------------------------
# Decoding (use stored logits once; reorder class axis; log-softmax)
# ---------------------------
lm_results = {
    "session": [],
    "block": [],
    "trial": [],
    "true_sentence": [],
    "pred_sentence": [],
}

with torch.no_grad():
    with tqdm(total=total_trials, desc="Decoding", unit="trial") as pbar:
        for session, data in test_data.items():
            for trial in range(len(data["logits_np"])):
                logits_np = data["logits_np"][trial]  # (T, C)
                logits_np = logits_np[:, order]  # 類別重排：blank → index 0
                logp = numpy_log_softmax(logits_np, axis=-1)

                try:
                    pred_sentence = decoder.decode(logp, beam_width=args.beam_width)
                except TypeError:
                    pred_sentence = decoder.decode(logp)

                lm_results["session"].append(session)
                lm_results["block"].append(data["block_num"][trial])
                lm_results["trial"].append(data["trial_num"][trial])
                lm_results["true_sentence"].append(
                    data["sentence_label"][trial] if args.eval_type == "val" else None
                )
                lm_results["pred_sentence"].append(pred_sentence)
                pbar.update(1)

# ---------------------------
# Compute WER (if validation set)
# ---------------------------
if args.eval_type == "val":
    total_words, total_edits = 0, 0
    for i in range(len(lm_results["pred_sentence"])):
        true_sentence = remove_punctuation(lm_results["true_sentence"][i]).strip()
        pred_sentence = remove_punctuation(lm_results["pred_sentence"][i]).strip()
        ed = editdistance.eval(true_sentence.split(), pred_sentence.split())
        total_words += len(true_sentence.split())
        total_edits += ed
        print(
            f"Trial {i}: WER = {ed}/{len(true_sentence.split())} = {ed/len(true_sentence.split()):.2f}"
        )
    if total_words > 0:
        print(f"Aggregate WER: {100 * total_edits / total_words:.2f}%")
    else:
        print("No reference words found to compute WER.")
else:
    print("[INFO] Test mode: skipping WER computation.")

# ---------------------------
# Save results
# ---------------------------
out_csv = os.path.join(
    args.model_path,
    f"baseline_rnn_{args.eval_type}_kenlm_{time.strftime('%Y%m%d_%H%M%S')}.csv",
)
pd.DataFrame(
    {"id": range(len(lm_results["pred_sentence"])), "text": lm_results["pred_sentence"]}
).to_csv(out_csv, index=False)
print(f"[INFO] Results saved to {out_csv}")
