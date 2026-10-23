"""
Detector accuracy check on a small hand-labelled set.

Calls the REAL Claim Detector (live Groq) on 20 claims whose ground truth is
unambiguous, then reports a confusion matrix, precision, recall and the routing
decision for each. It needs only GROQ_API_KEY - no Neo4j, no API server.

Be clear about what this is: 20 hand-picked claims, labelled by the author. It
shows the detector separates obvious misinformation from obvious non-
misinformation. It is NOT a benchmark and says nothing about subtle claims.

Run from backend/:

    python tests/eval_detector.py
"""

import json
import os
import sys
import time

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND)
os.environ.setdefault("CHROMA_PATH", os.path.join(BACKEND, "chroma_db"))

from agents.claim_detector import detect_claim   # noqa: E402

# (text, is_misinformation)
LABELLED = [
    # --- misinformation -------------------------------------------------------
    ("Drinking bleach cures COVID-19, doctors are hiding it from you.", True),
    ("5G towers spread the coronavirus and are being switched on to depopulate cities.", True),
    ("The MMR vaccine causes autism in children.", True),
    ("The 2020 US presidential election was stolen through millions of fake ballots, proven by leaked documents.", True),
    ("NASA has confirmed a giant asteroid will wipe out Earth next month and is covering it up.", True),
    ("Vaccines contain microchips that Bill Gates uses to track people.", True),
    ("Chemtrails sprayed by planes are being used to control the weather and poison the population.", True),
    ("Eating raw garlic cures cancer, so chemotherapy is unnecessary.", True),
    ("The Apollo moon landings were filmed in a Hollywood studio; humans never went to the moon.", True),
    ("The central bank will freeze all bank accounts tomorrow at midnight, withdraw your cash now.", True),
    # --- not misinformation ---------------------------------------------------
    ("The Earth orbits the Sun once approximately every 365 days.", False),
    ("Water boils at 100 degrees Celsius at sea level.", False),
    ("The city council voted on Tuesday to approve the new road maintenance budget.", False),
    ("The World Health Organization recommends adults get at least 150 minutes of moderate exercise per week.", False),
    ("Snopes reviewed the viral claim that 5G causes COVID-19 and rated it false.", False),
    ("In my opinion the new phone's camera is better than last year's model.", False),
    ("Paris is the capital of France.", False),
    ("The measles vaccine is safe and effective at preventing measles, according to the CDC.", False),
    ("Scientists published a study showing regular exercise lowers the risk of heart disease.", False),
    ("The football team won the championship match 2-1 on Sunday.", False),
]


def main() -> int:
    rows, tp, fp, tn, fn = [], 0, 0, 0, 0
    t0 = time.time()
    for text, truth in LABELLED:
        r = detect_claim(text)
        pred = r.is_misinformation
        tp += truth and pred
        fp += (not truth) and pred
        tn += (not truth) and (not pred)
        fn += truth and (not pred)
        rows.append({"claim": text, "truth": truth, "predicted": pred,
                     "correct": truth == pred, "confidence": r.confidence,
                     "severity": r.severity, "status": r.status, "category": r.category,
                     "rationale": r.rationale, "techniques": r.techniques})

    n = len(LABELLED)
    acc = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"\n{'CLAIM':<66} {'TRUTH':>6} {'PRED':>6} {'CONF':>5} {'SEV':>4}  STATUS")
    print("-" * 110)
    for x in rows:
        flag = "" if x["correct"] else "   <-- WRONG"
        print(f"{x['claim'][:64]:<66} {str(x['truth']):>6} {str(x['predicted']):>6} "
              f"{x['confidence']:>5.2f} {x['severity']:>4}  {x['status']}{flag}")

    mis = [x for x in rows if x["truth"]]
    ok = [x for x in rows if not x["truth"]]
    print("\nConfusion matrix (positive = misinformation)")
    print(f"    TP={tp}  FN={fn}\n    FP={fp}  TN={tn}")
    print(f"\n  accuracy  {acc:.0%}  ({tp + tn}/{n})")
    print(f"  precision {precision:.0%}")
    print(f"  recall    {recall:.0%}")
    print(f"  F1        {f1:.2f}")
    print(f"\n  mean severity - misinformation: {sum(x['severity'] for x in mis)/len(mis):.1f}, "
          f"non-misinformation: {sum(x['severity'] for x in ok)/len(ok):.1f}")
    print(f"  escalated to human review: {sum(x['status']=='review_required' for x in mis)}/{len(mis)} "
          f"misinformation, {sum(x['status']=='review_required' for x in ok)}/{len(ok)} non-misinformation")
    print(f"  dropped (not analysed further): {sum(x['status']=='dropped' for x in ok)}/{len(ok)} "
          f"non-misinformation, {sum(x['status']=='dropped' for x in mis)}/{len(mis)} misinformation")
    print(f"  ({time.time() - t0:.0f}s)")

    out_dir = os.path.join(BACKEND, "tests", "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "detector_eval.json"), "w", encoding="utf-8") as fh:
        json.dump({"n": n, "accuracy": acc, "precision": precision, "recall": recall, "f1": f1,
                   "tp": tp, "fp": fp, "tn": tn, "fn": fn, "rows": rows}, fh, indent=2,
                  ensure_ascii=False)
    print(f"\n  saved -> tests/results/detector_eval.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
