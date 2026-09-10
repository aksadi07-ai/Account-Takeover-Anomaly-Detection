from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

OUT=Path("reports"); FIG=OUT/"figures"

def main():
    OUT.mkdir(exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
    paths=[
        ("reports/isolation_forest_scores.csv","Isolation Forest","isolation_forest_score"),
        ("reports/autoencoder_scores.csv","Autoencoder","autoencoder_score"),
        ("reports/lstm_autoencoder_scores.csv","LSTM Autoencoder","lstm_autoencoder_score")
    ]
    combined=None; rows=[]

    plt.figure(figsize=(8,5))
    for path,name,score_col in paths:
        p=Path(path)
        if not p.exists(): continue
        d=pd.read_csv(p)
        if combined is None:
            combined=d[["session_id","user_id","label","attack_type"]].copy()
        combined[name]=d[score_col]
        precision,recall,_=precision_recall_curve(d.label,d[score_col])
        ap=average_precision_score(d.label,d[score_col])
        plt.plot(recall,precision,label=f"{name} (AP={ap:.3f})")
        rows.append({"model":name,"pr_auc":ap})

    if combined is None:
        raise FileNotFoundError("No score files found.")

    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Held-out Test Set — Precision-Recall Comparison")
    plt.legend(); plt.tight_layout()
    plt.savefig(FIG/"model_comparison_pr_curve.png",dpi=160); plt.close()

    pd.DataFrame(rows).to_csv(OUT/"results.csv",index=False)
    combined.to_csv(OUT/"combined_scores.csv",index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nImportant: thresholds are selected on validation data; metrics are reported on the held-out test set.")

if __name__=="__main__":
    main()
