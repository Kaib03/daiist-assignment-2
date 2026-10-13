"""
Assignment 2 dashboard. Phone use detection for a delivery fleet's in-cab camera.

    uv run python main.py app

Loads weights/best.pt and the files train.py wrote to artifacts/. Nothing is trained here.
"""

import json
import os
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
WEIGHTS = ROOT / "weights" / "best.pt"

S = json.loads((ART / "summary.json").read_text())
NAMES = S["dataset"]["names"]
COSTS = S["costs"]
PHONE_CLASSES = [1, 2]
IMGSZ = S["training"]["imgsz"]

model = YOLO(str(WEIGHTS))
valid = pd.read_csv(ART / "scores_valid.csv")
test = pd.read_csv(ART / "scores_test.csv")
sweep = pd.read_csv(ART / "cost_sweep.csv")
per_class = pd.read_csv(ART / "per_class_test.csv")
class_counts = pd.read_csv(ART / "class_counts.csv", index_col=0)
cm8 = pd.read_csv(ART / "image_confusion_8class.csv", index_col=0)
sens = pd.read_csv(ART / "sensitivity.csv")
EXAMPLES = sorted(str(p) for p in (ROOT / "examples").glob("*.jpg"))

LO0, HI0 = S["band"]["lo"], S["band"]["hi"]


# ---- decision maths, same formulas as train.py ---------------------------

def outcome(score: float, lo: float, hi: float) -> str:
    if score >= hi:
        return "FAIL, automatic coaching alert"
    if score >= lo:
        return "REVIEW, send to a safety analyst"
    return "PASS, no action"


def band_cost(df, lo, hi, base_rate, c_fn, c_fp, c_review):
    y, s = df.true_phone.values, df.phone_score.values
    def split(mask):
        sc = s[mask]
        fail, passed = (sc >= hi).mean(), (sc < lo).mean()
        return fail, 1 - fail - passed, passed
    pf, pr, pp = split(y == 1)
    nf, nr, npass = split(y == 0)
    pos = pf * COSTS["C_TP"] + pr * (c_review + COSTS["C_TP"]) + pp * c_fn
    neg = nf * c_fp + nr * c_review + npass * COSTS["C_TN"]
    cost = 1000 * (base_rate * pos + (1 - base_rate) * neg)
    review_share = base_rate * pr + (1 - base_rate) * nr
    return cost, review_share


def people(review_share):
    return S["frames_per_day"] * review_share * S["review_seconds"] / 3600 / S["analyst_hours"]


# ---- tab 1, single image --------------------------------------------------

def predict(image, lo, hi):
    if image is None:
        return None, "Upload a dashcam frame, or pick an example.", None
    # Gradio gives RGB, Ultralytics reads a numpy array as BGR like OpenCV
    r = model.predict(np.ascontiguousarray(image[..., ::-1]), conf=0.001, imgsz=IMGSZ, device="cpu", verbose=False)[0]
    best = np.zeros(len(NAMES))
    for c, s in zip(r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
        best[c] = max(best[c], s)
    phone = float(best[PHONE_CLASSES].max())
    keep = r.boxes.conf >= max(lo, 0.05)
    r.boxes = r.boxes[keep][:3]
    drawn = r.plot()[:, :, ::-1]
    table = pd.DataFrame({"class": NAMES, "best confidence": np.round(best, 3)}).sort_values(
        "best confidence", ascending=False)
    text = (f"### {outcome(phone, lo, hi)}\n"
            f"Phone score {phone:.3f}, the stronger of the texting and calling boxes. "
            f"Pass below {lo:.2f}, review between {lo:.2f} and {hi:.2f}, fail at {hi:.2f} or above.")
    return drawn, text, table


# ---- tab 2, operating point -----------------------------------------------

def operating_point(lo, hi, base_rate, c_fn, c_fp, c_review, split):
    if lo > hi:
        lo, hi = hi, lo
    df = test if split == "test" else valid
    y, s = df.true_phone.values, df.phone_score.values
    bucket = np.where(s >= hi, "fail", np.where(s >= lo, "review", "pass"))
    three = pd.crosstab(pd.Series(np.where(y == 1, "phone", "no phone"), name="truth"),
                        pd.Series(bucket, name="decision")).reindex(
        index=["phone", "no phone"], columns=["pass", "review", "fail"], fill_value=0)
    # After review the analyst decides correctly, so the final calls are:
    final_pos = (bucket == "fail") | ((bucket == "review") & (y == 1))
    cm = pd.DataFrame([[int((final_pos & (y == 1)).sum()), int((~final_pos & (y == 1)).sum())],
                       [int((final_pos & (y == 0)).sum()), int((~final_pos & (y == 0)).sum())]],
                      index=["truth phone", "truth no phone"], columns=["flagged", "not flagged"])
    cost, review_share = band_cost(df, lo, hi, base_rate, c_fn, c_fp, c_review)
    alternatives = pd.DataFrame([
        ["do nothing", 1000 * base_rate * c_fn, 0.0],
        ["person checks every frame", 1000 * (c_review + base_rate * COSTS["C_TP"]), 1.0],
        [f"model alone at {S['single_threshold']:.2f}",
         *band_cost(df, S["single_threshold"], S["single_threshold"], base_rate, c_fn, c_fp, c_review)],
        ["your band", cost, review_share],
    ], columns=["policy", "euros per 1,000 frames", "share reviewed"])
    alternatives["analysts needed"] = alternatives["share reviewed"].map(people).round(1)
    alternatives["euros per 1,000 frames"] = alternatives["euros per 1,000 frames"].round(0)
    alternatives["share reviewed"] = (100 * alternatives["share reviewed"]).round(1).astype(str) + "%"
    headline = (f"## {cost:,.0f} euros per 1,000 frames\n"
                f"**{100 * review_share:.1f}%** of frames go to a person at a {100 * base_rate:.0f}% base rate. "
                f"That is **{S['frames_per_day'] * review_share:,.0f} frames a day**, "
                f"about **{people(review_share):.1f} analysts** at {S['review_seconds']} seconds each. "
                f"Counts below are the {len(df)} images of the {split} split, the cost is reweighted to the base rate.")
    return headline, three.reset_index(), cm.reset_index(), alternatives, cost_curve(lo, hi, base_rate, c_fn, c_fp, c_review)


def cost_curve(lo, hi, base_rate, c_fn, c_fp, c_review):
    grid = sweep.threshold.values
    tuned = [band_cost(test, t, t, base_rate, c_fn, c_fp, c_review)[0] for t in grid]
    tuned_v = [band_cost(valid, t, t, base_rate, c_fn, c_fp, c_review)[0] for t in grid]
    coco = [band_cost(test.assign(phone_score=test.coco_phone_score), t, t, base_rate, c_fn, c_fp, c_review)[0]
            for t in grid]
    fig = go.Figure()
    fig.add_scatter(x=grid, y=tuned, name="finetuned, test", line=dict(color="#2563eb", width=3))
    fig.add_scatter(x=grid, y=tuned_v, name="finetuned, valid", line=dict(color="#2563eb", dash="dot"))
    fig.add_scatter(x=grid, y=coco, name="COCO zero shot, test", line=dict(color="#9ca3af"))
    fig.add_hline(y=1000 * base_rate * c_fn, line_dash="dash", line_color="#dc2626",
                  annotation_text="do nothing")
    t_star = S["single_threshold"]
    fig.add_scatter(x=[t_star], y=[band_cost(test, t_star, t_star, base_rate, c_fn, c_fp, c_review)[0]],
                    mode="markers", marker=dict(size=13, color="#f59e0b", symbol="star"),
                    name=f"model alone minimum on valid, {t_star:.2f}")
    fig.add_vrect(x0=lo, x1=hi, fillcolor="#f59e0b", opacity=0.12, line_width=0,
                  annotation_text="review band", annotation_position="top left")
    fig.update_layout(title="Expected cost when the model decides alone, by threshold",
                      xaxis_title="confidence threshold on the phone score",
                      yaxis_title="euros per 1,000 frames", height=430, legend=dict(orientation="h", y=-0.25),
                      margin=dict(t=50, b=10))
    return fig


# ---- tab 3 and 4 ----------------------------------------------------------

def per_class_fig():
    long = per_class.melt(id_vars="class", value_vars=["precision", "recall", "AP50"])
    fig = px.bar(long, x="class", y="value", color="variable", barmode="group", height=420,
                 title="Per class results on the test split, detection metrics")
    fig.update_layout(xaxis_tickangle=-25, yaxis_range=[0, 1], margin=dict(t=50))
    return fig


def cm8_fig():
    order = [n for n in NAMES if n in cm8.index]
    cols = [n for n in NAMES + ["nothing"] if n in cm8.columns]
    m = cm8.reindex(index=order, columns=cols, fill_value=0)
    fig = px.imshow(m.values, x=[c.split(" - ")[-1] for c in cols], y=[c.split(" - ")[-1] for c in order],
                    text_auto=True, color_continuous_scale="Blues", height=480,
                    labels=dict(x="predicted, top box", y="true class", color="images"),
                    title="Image level confusion matrix on test")
    return fig


def counts_fig():
    long = class_counts.reset_index(names="class").melt(id_vars="class", var_name="split", value_name="instances")
    fig = px.bar(long, x="class", y="instances", color="split", barmode="group", height=420,
                 title="Instances per class and split")
    fig.update_layout(xaxis_tickangle=-25, margin=dict(t=50))
    return fig


def score_hist():
    df = test.assign(truth=np.where(test.true_phone == 1, "phone", "no phone"))
    fig = px.histogram(df, x="phone_score", color="truth", nbins=40, barmode="overlay", height=380,
                       title="Phone score on test images, by truth", opacity=0.7)
    fig.add_vline(x=LO0, line_dash="dot"); fig.add_vline(x=HI0, line_dash="dot")
    return fig


# ---- layout ---------------------------------------------------------------

with gr.Blocks(title="Phone use review desk") as demo:
    band_text = (f"pass below {LO0:.2f}, fail from {HI0:.2f}" if HI0 <= 1
                 else f"pass below {LO0:.2f}, everything else goes to review, no automatic fail")
    gr.Markdown(
        "# Phone use review desk\n"
        "A fleet's cabin camera samples one frame per van every 10 minutes. The model scores each frame for phone use, "
        "then it **passes**, goes to **human review**, or **fails** and triggers a coaching alert. "
        f"Costs per frame, missed phone use {COSTS['C_FN']:.0f} euros, false alert {COSTS['C_FP']:.0f}, "
        f"coaching {COSTS['C_TP']:.0f}, review {COSTS['C_REVIEW']:.2f}. Band chosen on the valid split, "
        f"**{band_text}**, the cheapest band that {S.get('max_analysts', 2)} analysts can keep up with.")
    with gr.Row():
        lo = gr.Slider(0, 1.01, value=LO0, step=0.01, label="Lower threshold, below this the frame passes")
        hi = gr.Slider(0, 1.01, value=HI0, step=0.01, label="Upper threshold, at or above this the frame fails, 1.01 means never")

    with gr.Tab("Operating point"):
        with gr.Row():
            base_rate = gr.Slider(0.005, 0.3, value=S["base_rate"], step=0.005, label="Base rate of phone use")
            c_fn = gr.Slider(1, 200, value=COSTS["C_FN"], step=1, label="Cost of a missed phone use, euros")
            c_fp = gr.Slider(0, 100, value=COSTS["C_FP"], step=1, label="Cost of a false alert, euros")
            c_review = gr.Slider(0.05, 3, value=COSTS["C_REVIEW"], step=0.05, label="Cost of one review, euros")
            split = gr.Radio(["test", "valid"], value="test", label="Split")
        headline = gr.Markdown()
        with gr.Row():
            three = gr.Dataframe(label="Where frames go, pass, review or fail")
            cm = gr.Dataframe(label="Confusion matrix after review, analysts assumed correct")
        alternatives = gr.Dataframe(label="Your band against the alternatives")
        curve = gr.Plot()
        gr.Plot(score_hist())
        gr.Markdown("**Sensitivity**, best band picked on valid when one assumption moves, cost on test")
        gr.Dataframe(sens)

    with gr.Tab("Try an image"):
        with gr.Row():
            img = gr.Image(type="numpy", label="Dashcam frame")
            out_img = gr.Image(label="Top boxes above the lower threshold")
        verdict = gr.Markdown()
        scores = gr.Dataframe(label="Best confidence per class")
        if EXAMPLES:
            gr.Examples(EXAMPLES, inputs=img)
        gr.Button("Score frame", variant="primary").click(predict, [img, lo, hi], [out_img, verdict, scores])
        img.change(predict, [img, lo, hi], [out_img, verdict, scores])

    with gr.Tab("Per class results"):
        d = S["detection_test"]["finetuned"]
        gr.Markdown(f"Overall on test, AP50 **{d['AP50']}**, mAP50-95 **{d['mAP50-95']}**. "
                    f"Image level 8 class accuracy **{S['image_level_8class_accuracy_test']}** against "
                    f"**{S['majority_class_accuracy_test']}** for always guessing the biggest class. "
                    "The positive class for the decision is texting or talking on the phone.")
        gr.Dataframe(per_class)
        gr.Plot(per_class_fig())
        gr.Plot(cm8_fig())
        gr.Dataframe(pd.DataFrame(S["phone_decision_test"]).T.reset_index(names="model"),
                     label="Phone decision on test at each model's own cost minimising threshold")

    with gr.Tab("Dataset"):
        ds = S["dataset"]
        gr.Markdown(f"Roboflow Universe **{ds['workspace']}/{ds['project']}** version **{ds['version']}**. "
                    f"Images per split {ds['split_images']}. One box per image, box covers "
                    f"{ds['box_area_share']['median']:.0%} of the frame at the median.")
        gr.Plot(counts_fig())

    inputs = [lo, hi, base_rate, c_fn, c_fp, c_review, split]
    for c in inputs:
        c.change(operating_point, inputs, [headline, three, cm, alternatives, curve])
    demo.load(operating_point, inputs, [headline, three, cm, alternatives, curve])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=int(os.environ.get("APP_PORT", "7860")))
