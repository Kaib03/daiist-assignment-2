"""
Assignment 2 training pipeline. Phone use detection for a delivery fleet's in-cab camera.

    uv run python main.py train

Start to finish, with no manual steps, this script
1. downloads the dataset version from Roboflow Universe,
2. describes it (instances per class, split sizes, box size, duplicates across splits),
3. finetunes yolo11n from the COCO checkpoint and copies the best weights to weights/,
4. evaluates on the test split with per class numbers and the confusion matrix,
5. scores every valid and test image for "is the driver using a phone",
   for the finetuned model and for the COCO model used zero shot,
6. prices the decision, sweeps the threshold, and picks the two thresholds
   that split frames into pass, human review and fail. Thresholds are picked on
   the valid split and then reported on the test split.

Everything the dashboard needs is written to artifacts/.

Run with --eval-only to skip the finetune and reuse weights/best.pt. That is
only a shortcut for iterating on the evaluation, the default run trains.
"""

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import numpy as np
import pandas as pd
from PIL import Image

from api_key import load_api_key

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets" / "driver"
WEIGHTS = ROOT / "weights" / "best.pt"
ART = ROOT / "artifacts"

# Dataset, pinned so anyone can rerun on exactly the same data
WORKSPACE, PROJECT, VERSION = "yzc", "distracted-driver-detection-fxeff", 3

# Training settings. CPU only, see REPORT.md.
MODEL = "yolo11n.pt"
EPOCHS = 30
IMGSZ = 320          # one box per image covering ~60% of the frame, so low resolution loses little
BATCH = 16
PATIENCE = 10
SEED = 0

# The decision. Positive = the driver is using a phone, texting or on a call.
PHONE_CLASSES = [1, 2]
COCO_CELL_PHONE = 67

# Costs in euros per frame, justified in REPORT.md
C_FN = 40.0      # phone use missed, driver not coached, expected crash cost carried forward
C_FP = 15.0      # automatic alert sent about a driver who was not on the phone
C_TP = 10.0      # coaching session for a driver who was on the phone
C_TN = 0.0
C_REVIEW = 0.25  # 30 seconds of a safety analyst at 30 euros an hour

BASE_RATE = 0.05          # share of sampled frames in production that really show phone use
FRAMES_PER_DAY = 14_400   # 300 vans, one frame every 10 minutes over an 8 hour shift
REVIEW_SECONDS = 30
ANALYST_HOURS = 7         # productive review hours per analyst per day
MAX_ANALYSTS = 2          # what a 300 van fleet would realistically staff for this

# Thresholds swept. 1.01 means never: with lo = 1.01 nothing is reviewed or failed, with hi = 1.01 nothing auto fails
GRID = np.append(np.round(np.arange(0.0, 1.0001, 0.01), 2), 1.01)


# 1. Data -----------------------------------------------------------------

def download() -> Path:
    if (DATA_DIR / "data.yaml").exists():
        print(f"dataset already at {DATA_DIR}")
    else:
        from roboflow import Roboflow
        rf = Roboflow(api_key=load_api_key())
        rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download(
            "yolov11", location=str(DATA_DIR))
    # Roboflow writes relative paths that depend on the working directory, make them absolute
    yaml_path = DATA_DIR / "data.yaml"
    text = yaml_path.read_text()
    for split, folder in [("train", "train"), ("val", "valid"), ("test", "test")]:
        lines = [l for l in text.splitlines() if not l.startswith(f"{split}:")]
        lines.insert(0, f"{split}: {DATA_DIR / folder / 'images'}")
        text = "\n".join(lines)
    yaml_path.write_text(text + "\n")
    return yaml_path


def class_names() -> list[str]:
    import yaml
    return yaml.safe_load((DATA_DIR / "data.yaml").read_text())["names"]


def read_labels(split: str) -> pd.DataFrame:
    rows = []
    for img in sorted((DATA_DIR / split / "images").iterdir()):
        lab = DATA_DIR / split / "labels" / (img.stem + ".txt")
        lines = [l.split() for l in lab.read_text().strip().splitlines() if l.strip()] if lab.exists() else []
        for l in lines:
            rows.append({"split": split, "image": img.name, "cls": int(l[0]),
                         "w": float(l[3]), "h": float(l[4])})
        if not lines:
            rows.append({"split": split, "image": img.name, "cls": -1, "w": 0.0, "h": 0.0})
    return pd.DataFrame(rows)


def ahash(path: Path) -> str:
    a = np.asarray(Image.open(path).convert("L").resize((16, 12)), dtype=float)
    return "".join("1" if x else "0" for x in (a > a.mean()).flatten())


def describe(names: list[str]) -> dict:
    labels = pd.concat([read_labels(s) for s in ["train", "valid", "test"]])
    labels["area"] = labels["w"] * labels["h"]
    counts = labels.pivot_table(index="cls", columns="split", values="image", aggfunc="count").fillna(0).astype(int)
    counts = counts[["train", "valid", "test"]]
    counts.index = [names[i] if i >= 0 else "no label" for i in counts.index]

    per_img = labels.groupby(["split", "image"]).size()
    hashes = {}
    for split in ["train", "valid", "test"]:
        for img in (DATA_DIR / split / "images").iterdir():
            hashes.setdefault(ahash(img), set()).add(split)
    dup_test_train = sum(1 for s in hashes.values() if "test" in s and "train" in s)

    stats = {
        "split_images": {s: int(labels[labels.split == s].image.nunique()) for s in ["train", "valid", "test"]},
        "instances": counts.to_dict(orient="index"),
        "boxes_per_image": {str(k): int(v) for k, v in Counter(per_img.values).items()},
        "box_area_share": {q: round(float(labels.area.quantile(p)), 3)
                           for q, p in [("min", 0), ("p10", .1), ("median", .5), ("p90", .9), ("max", 1)]},
        "test_images_with_near_duplicate_in_train": int(dup_test_train),
        "phone_share": {s: round(float(labels[labels.split == s].cls.isin(PHONE_CLASSES).mean()), 3)
                        for s in ["train", "valid", "test"]},
    }
    print("\ninstances per class\n", counts)
    print(json.dumps({k: v for k, v in stats.items() if k != "instances"}, indent=1))
    counts.to_csv(ART / "class_counts.csv")
    labels.to_csv(ART / "labels.csv", index=False)
    return stats


# 2. Finetune -------------------------------------------------------------

def finetune(yaml_path: Path) -> None:
    from ultralytics import YOLO
    model = YOLO(MODEL)
    model.train(data=str(yaml_path), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, patience=PATIENCE,
                device="cpu", workers=2, seed=SEED, deterministic=True,
                project=str(ROOT / "runs"), name="driver", exist_ok=True)
    best = Path(model.trainer.best)
    WEIGHTS.parent.mkdir(exist_ok=True)
    shutil.copy(best, WEIGHTS)
    shutil.copy(best.parent.parent / "results.csv", ART / "training_curve.csv")
    print(f"copied {best} -> {WEIGHTS}")


# 3. Detection metrics on test --------------------------------------------

def detection_metrics(yaml_path: Path, names: list[str]) -> dict:
    from ultralytics import YOLO
    model = YOLO(str(WEIGHTS))
    res = model.val(data=str(yaml_path), split="test", imgsz=IMGSZ, device="cpu", batch=BATCH,
                    project=str(ROOT / "runs"), name="test_eval", exist_ok=True, plots=True)
    p, r, ap50, ap = res.box.p, res.box.r, res.box.ap50, res.box.ap
    idx = list(res.box.ap_class_index)
    per_class = pd.DataFrame({
        "class": [names[i] for i in idx],
        "instances": [int((read_labels("test").cls == i).sum()) for i in idx],
        "precision": np.round(p, 3), "recall": np.round(r, 3),
        "AP50": np.round(ap50, 3), "mAP50-95": np.round(ap, 3),
    })
    per_class.to_csv(ART / "per_class_test.csv", index=False)
    cm = res.confusion_matrix.matrix  # rows predicted, cols true, last index = background
    np.save(ART / "detection_confusion.npy", cm)
    for f in ["confusion_matrix.png", "confusion_matrix_normalized.png"]:
        src = ROOT / "runs" / "test_eval" / f
        if src.exists():
            shutil.copy(src, ART / f"detection_{f}")
    overall = {"AP50": round(float(res.box.map50), 3), "mAP50-95": round(float(res.box.map), 3),
               "precision": round(float(np.mean(p)), 3), "recall": round(float(np.mean(r)), 3)}
    print("\nper class on test\n", per_class.to_string(index=False), "\noverall", overall)
    return overall


# 4. Image level scores ---------------------------------------------------

def image_scores(split: str, names: list[str]) -> pd.DataFrame:
    """For every image, the finetuned model's best confidence per class and the COCO phone score."""
    from ultralytics import YOLO
    tuned, coco = YOLO(str(WEIGHTS)), YOLO(MODEL)
    labels = read_labels(split).groupby("image").cls.first()
    rows = []
    for img in sorted((DATA_DIR / split / "images").iterdir()):
        r = tuned.predict(str(img), conf=0.001, imgsz=IMGSZ, device="cpu", verbose=False)[0]
        best = np.zeros(len(names))
        for c, s in zip(r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
            best[c] = max(best[c], s)
        rc = coco.predict(str(img), conf=0.001, classes=[COCO_CELL_PHONE], device="cpu", verbose=False)[0]
        coco_phone = float(rc.boxes.conf.max()) if len(rc.boxes) else 0.0
        true_cls = int(labels[img.name])
        rows.append({"image": img.name, "true_cls": true_cls, "true_phone": int(true_cls in PHONE_CLASSES),
                     **{f"s{i}": round(float(v), 4) for i, v in enumerate(best)},
                     # P(phone) proxy: the strongest texting or calling box
                     "phone_score": round(float(best[PHONE_CLASSES].max()), 4),
                     "pred_cls": int(best.argmax()) if best.max() > 0 else -1,
                     "coco_phone_score": round(coco_phone, 4)})
    df = pd.DataFrame(rows)
    df.to_csv(ART / f"scores_{split}.csv", index=False)
    return df


# 5. Costing --------------------------------------------------------------

def rates(y: np.ndarray, s: np.ndarray, lo: float, hi: float) -> dict:
    """Share of positives and negatives that land in pass (s < lo), review, fail (s >= hi)."""
    out = {}
    for name, mask in [("pos", y == 1), ("neg", y == 0)]:
        sc = s[mask]
        fail = float((sc >= hi).mean())
        passed = float((sc < lo).mean())
        out[name] = {"fail": fail, "review": 1 - fail - passed, "pass": passed}
    return out


def cost_per_1000(y, s, lo, hi, base_rate=BASE_RATE, c_fn=C_FN, c_fp=C_FP, c_review=C_REVIEW) -> dict:
    """Expected euros per 1,000 frames at the production base rate, not the dataset's mix.
    Reviewed frames are assumed to be decided correctly by the analyst."""
    r = rates(y, s, lo, hi)
    pos = r["pos"]["fail"] * C_TP + r["pos"]["review"] * (c_review + C_TP) + r["pos"]["pass"] * c_fn
    neg = r["neg"]["fail"] * c_fp + r["neg"]["review"] * c_review + r["neg"]["pass"] * C_TN
    review_share = base_rate * r["pos"]["review"] + (1 - base_rate) * r["neg"]["review"]
    return {"cost": 1000 * (base_rate * pos + (1 - base_rate) * neg), "review_share": review_share,
            "missed_share": base_rate * r["pos"]["pass"], "false_alert_share": (1 - base_rate) * r["neg"]["fail"]}


def band_matrix(y, s, smooth=True, base_rate=BASE_RATE, c_fn=C_FN, c_fp=C_FP, c_review=C_REVIEW):
    """Cost and review share for every (lo, hi) pair on GRID at once. Row = lo, column = hi.
    Same formula as cost_per_1000, written with the share of scores at or above each threshold.

    smooth=True is used only when picking thresholds. The share of images above a threshold
    inside the score range gets a Jeffreys prior, (count + 0.5) / (n + 1), so an empty tail on
    275 valid negatives counts as half an image and not as proof that the rate is zero.
    lo = 0 (nothing passes) and hi = NEVER (nothing fails) stay exact."""
    ge = (s[None, :] >= GRID[:, None])
    def share(mask):
        k, n = ge[:, mask].sum(axis=1).astype(float), mask.sum()
        out = (k + 0.5) / (n + 1) if smooth else k / n
        out[GRID == 0] = 1.0
        out[GRID > 1] = 0.0
        return out
    p_ge, n_ge = share(y == 1), share(y == 0)
    pf, pr, pp = p_ge[None, :], p_ge[:, None] - p_ge[None, :], 1 - p_ge[:, None]
    nf, nr = n_ge[None, :], n_ge[:, None] - n_ge[None, :]
    pos = pf * C_TP + pr * (c_review + C_TP) + pp * c_fn
    neg = nf * c_fp + nr * c_review
    cost = 1000 * (base_rate * pos + (1 - base_rate) * neg)
    review = base_rate * pr + (1 - base_rate) * nr
    ok = GRID[None, :] >= GRID[:, None]
    return np.where(ok, cost, np.inf), np.where(ok, review, np.inf)


def best_band(y, s, max_review_share=1.0, smooth=True, single=False, **kw) -> tuple[float, float, float]:
    """Cheapest (lo, hi) pair on GRID among bands within the review capacity.
    single=True restricts to lo == hi, the model deciding alone."""
    cost, review = band_matrix(y, s, smooth=smooth, **kw)
    cost = np.where(review <= max_review_share + 1e-12, cost, np.inf)
    if single:
        cost = np.where(np.eye(len(GRID), dtype=bool), cost, np.inf)
    i, j = np.unravel_index(np.argmin(cost), cost.shape)
    return float(cost[i, j]), float(GRID[i]), float(GRID[j])


def best_single(y, s, **kw) -> tuple[float, float]:
    c, t, _ = best_band(y, s, single=True, **kw)
    return c, t


def people_needed(review_share: float) -> float:
    return FRAMES_PER_DAY * review_share * REVIEW_SECONDS / 3600 / ANALYST_HOURS


def costing(valid: pd.DataFrame, test: pd.DataFrame) -> dict:
    yv, sv = valid.true_phone.values, valid.phone_score.values
    yt, st = test.true_phone.values, test.phone_score.values

    # Single threshold sweep, model decides alone, plotted in the dashboard
    sweep = pd.DataFrame([{
        "threshold": t,
        "cost_valid": cost_per_1000(yv, sv, t, t)["cost"],
        "cost_test": cost_per_1000(yt, st, t, t)["cost"],
        "cost_coco_test": cost_per_1000(yt, test.coco_phone_score.values, t, t)["cost"],
        "recall_test": float((st[yt == 1] >= t).mean()),
        "fpr_test": float((st[yt == 0] >= t).mean()),
    } for t in GRID])
    sweep.to_csv(ART / "cost_sweep.csv", index=False)

    # Every threshold is picked on valid, then reported on test exactly as observed
    cap = MAX_ANALYSTS * ANALYST_HOURS * 3600 / REVIEW_SECONDS / FRAMES_PER_DAY
    _, t_single = best_single(yv, sv)
    _, lo_naive, hi_naive = best_band(yv, sv, max_review_share=cap, smooth=False)  # raw rates, for comparison
    _, lo_free, hi_free = best_band(yv, sv)
    _, lo, hi = best_band(yv, sv, max_review_share=cap)                            # the band we ship
    _, t_coco = best_single(valid.true_phone.values, valid.coco_phone_score.values)

    band = cost_per_1000(yt, st, lo, hi)
    policies = {
        "do nothing (every frame passes)": {"cost": 1000 * BASE_RATE * C_FN, "review_share": 0.0},
        "person checks every frame": {"cost": 1000 * (C_REVIEW + BASE_RATE * C_TP), "review_share": 1.0},
        "COCO zero shot alone": cost_per_1000(yt, test.coco_phone_score.values, t_coco, t_coco),
        "finetuned model alone": cost_per_1000(yt, st, t_single, t_single),
        "finetuned model + band picked on raw valid rates": cost_per_1000(yt, st, lo_naive, hi_naive),
        "finetuned model + review band, no staffing limit": cost_per_1000(yt, st, lo_free, hi_free),
        f"finetuned model + review band, max {MAX_ANALYSTS} analysts": band,
    }
    for p in policies.values():
        p["people"] = people_needed(p["review_share"])
        p["cost"] = round(p["cost"], 1)

    # Sensitivity: how the chosen band and its cost move when the assumptions move
    sens = []
    for name, values, key in [("c_fn", [5, 10, 11, 12, 15, 20, 40, 80, 160], "c_fn"),
                              ("c_fp", [2, 5, 6, 7, 8, 9, 10, 15, 30, 60], "c_fp"),
                              ("base_rate", [0.01, 0.02, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.20], "base_rate"),
                              ("c_review", [0.10, 0.25, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 1.00, 2.00], "c_review")]:
        for v in values:
            c, l, h = best_band(yv, sv, max_review_share=cap, **{key: v})
            on_test = cost_per_1000(yt, st, l, h, **{key: v})
            sens.append({"assumption": name, "value": v, "lo": l, "hi": h,
                         "test_cost": round(on_test["cost"], 1), "review_share": round(on_test["review_share"], 3)})
    pd.DataFrame(sens).to_csv(ART / "sensitivity.csv", index=False)
    plot_cost_curve(sweep, t_single, lo, hi, yt, st)

    result = {
        "costs": {"C_FN": C_FN, "C_FP": C_FP, "C_TP": C_TP, "C_TN": C_TN, "C_REVIEW": C_REVIEW},
        "base_rate": BASE_RATE, "frames_per_day": FRAMES_PER_DAY,
        "review_seconds": REVIEW_SECONDS, "analyst_hours": ANALYST_HOURS,
        "single_threshold": t_single, "coco_threshold": t_coco, "band": {"lo": lo, "hi": hi},
        "band_unconstrained": {"lo": lo_free, "hi": hi_free}, "max_analysts": MAX_ANALYSTS,
        "band_raw_rates": {"lo": lo_naive, "hi": hi_naive},
        "max_review_share": cap,
        "band_on_test": band, "band_rates_test": rates(yt, st, lo, hi),
        "policies_test": policies,
    }
    print("\nchosen on valid: single", t_single, "band", lo, hi, "| raw rates", lo_naive, hi_naive,
          "| no staffing limit", lo_free, hi_free)
    print(pd.DataFrame(policies).T.round(3).to_string())
    return result


def plot_cost_curve(sweep: pd.DataFrame, t_single: float, lo: float, hi: float,
                    sweep_y_test: np.ndarray, sweep_s_test: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sw = sweep[sweep.threshold <= 1.0]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sw.threshold, sw.cost_valid, color="#2563eb", ls=":", label="finetuned, valid")
    ax.plot(sw.threshold, sw.cost_test, color="#2563eb", lw=2.5, label="finetuned, test")
    ax.plot(sw.threshold, sw.cost_coco_test, color="#9ca3af", label="COCO zero shot, test")
    ax.axhline(1000 * BASE_RATE * C_FN, color="#dc2626", ls="--", lw=1, label="do nothing")
    ax.axhline(1000 * (C_REVIEW + BASE_RATE * C_TP), color="#16a34a", ls="--", lw=1, label="person checks every frame")
    y_star = sw.loc[sw.threshold == t_single, "cost_valid"].iloc[0]
    ax.scatter([t_single], [y_star], s=140, marker="*", color="#f59e0b", zorder=5,
               label=f"model alone minimum on valid, {t_single:.2f}")
    band_test = cost_per_1000(sweep_y_test, sweep_s_test, lo, hi)["cost"]
    ax.axhline(band_test, color="#f59e0b", lw=2, label=f"chosen band on test, review from {lo:.2f}, no auto fail")
    ax.set_xlabel("confidence threshold on the phone score")
    ax.set_ylabel("euros per 1,000 frames")
    ax.set_title("Expected cost when the model decides alone, by threshold")
    ax.set_ylim(0, 2200); ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(0.98, 0.84)); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(ART / "cost_curve.png", dpi=130); plt.close(fig)


def binary_table(df: pd.DataFrame, col: str, t: float) -> dict:
    y, s = df.true_phone.values, df[col].values
    tp = int(((s >= t) & (y == 1)).sum()); fp = int(((s >= t) & (y == 0)).sum())
    fn = int(((s < t) & (y == 1)).sum()); tn = int(((s < t) & (y == 0)).sum())
    return {"threshold": t, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": round(tp / max(tp + fp, 1), 3), "recall": round(tp / max(tp + fn, 1), 3),
            "accuracy": round((tp + tn) / len(y), 3)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true", help="reuse weights/best.pt, skip the finetune")
    args = parser.parse_args()
    ART.mkdir(exist_ok=True)

    yaml_path = download()
    names = class_names()
    stats = describe(names)

    if args.eval_only and WEIGHTS.exists():
        print("skipping finetune, using", WEIGHTS)
    else:
        finetune(yaml_path)

    overall = detection_metrics(yaml_path, names)
    valid, test = image_scores("valid", names), image_scores("test", names)

    # 8 class confusion matrix at image level, top box per image
    cm8 = pd.crosstab(pd.Series([names[i] for i in test.true_cls], name="true"),
                      pd.Series([names[i] if i >= 0 else "nothing" for i in test.pred_cls], name="predicted"))
    cm8.to_csv(ART / "image_confusion_8class.csv")
    image_acc = float((test.true_cls == test.pred_cls).mean())

    cost = costing(valid, test)
    baselines = {
        "finetuned yolo11n": binary_table(test, "phone_score", cost["single_threshold"]),
        "COCO yolo11n zero shot (cell phone box)": binary_table(test, "coco_phone_score", cost["coco_threshold"]),
        "trivial, never flag": binary_table(test.assign(zero=0.0), "zero", 1.01),
    }
    print("\nphone decision on test\n", pd.DataFrame(baselines).T.to_string())

    summary = {
        "dataset": {"workspace": WORKSPACE, "project": PROJECT, "version": VERSION, "names": names, **stats},
        "training": {"model": MODEL, "epochs": EPOCHS, "imgsz": IMGSZ, "batch": BATCH,
                     "patience": PATIENCE, "seed": SEED, "device": "cpu"},
        "detection_test": {"finetuned": overall,
                           "coco_zero_shot": {"AP50": 0.0, "mAP50-95": 0.0,
                                              "note": "COCO has none of the 8 driver classes"},
                           "trivial_predict_nothing": {"AP50": 0.0, "mAP50-95": 0.0}},
        "image_level_8class_accuracy_test": round(image_acc, 3),
        # majority class taken from train, scored on test
        "majority_class_accuracy_test": round(float((test.true_cls == read_labels("train").cls.mode()[0]).mean()), 3),
        "phone_decision_test": baselines,
        **cost,
    }
    (ART / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {ART / 'summary.json'}")


if __name__ == "__main__":
    sys.exit(main())
