from pathlib import Path
import json, random
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from model_utils import ensure_dirs, threshold_from_validation, evaluate_scores

SEED=42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

SEQ=Path("data/processed/sequences.npz")
FEATURES=Path("data/processed/session_features.csv")

class LSTMAutoencoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=16, hidden_dim=32, latent_dim=16):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size+1, embedding_dim, padding_idx=0)
        self.encoder=nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.to_latent=nn.Linear(hidden_dim, latent_dim)
        self.from_latent=nn.Linear(latent_dim, hidden_dim)
        self.decoder=nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.output=nn.Linear(hidden_dim, embedding_dim)
    def forward(self,x):
        emb=self.embedding(x)
        _,(h,_)=self.encoder(emb)
        latent=self.to_latent(h[-1])
        hidden=torch.tanh(self.from_latent(latent)).unsqueeze(0)
        cell=torch.zeros_like(hidden)
        dec,_=self.decoder(emb,(hidden,cell))
        return self.output(dec)

def score_model(model,X,device):
    model.eval()
    with torch.no_grad():
        x=torch.tensor(X,dtype=torch.long,device=device)
        target=model.embedding(x)
        pred=model(x)
        err=((pred-target)**2).mean(dim=2)
        mask=(x!=0).float()
        return ((err*mask).sum(dim=1)/mask.sum(dim=1).clamp_min(1)).cpu().numpy()

def main():
    ensure_dirs()
    seq=np.load(SEQ)
    X=seq["sequences"]; y=seq["labels"]
    df=pd.read_csv(FEATURES)

    train_val,test=train_test_split(np.arange(len(X)),test_size=.20,stratify=y,random_state=42)
    train,val=train_test_split(train_val,test_size=.25,stratify=y[train_val],random_state=42)
    normal_train=train[y[train]==0]

    vocab_size=int(X.max())
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=LSTMAutoencoder(vocab_size).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    loss_fn=nn.MSELoss(reduction="none")
    x=torch.tensor(X[normal_train],dtype=torch.long,device=device)

    model.train()
    for epoch in range(70):
        opt.zero_grad()
        pred=model(x); target=model.embedding(x)
        per_pos=loss_fn(pred,target).mean(dim=2)
        mask=(x!=0).float()
        loss=(per_pos*mask).sum()/mask.sum().clamp_min(1)
        loss.backward(); opt.step()

    val_scores=score_model(model,X[val],device)
    threshold=threshold_from_validation(y[val],val_scores)
    test_scores=score_model(model,X[test],device)
    metrics=evaluate_scores(y[test],test_scores,threshold)

    torch.save({"state_dict":model.state_dict(),"vocab_size":vocab_size},"models/lstm_autoencoder.pt")
    Path("models/lstm_autoencoder_meta.json").write_text(json.dumps({
        "vocab_size":vocab_size,
        "threshold_selected_on":"validation",
        "threshold":threshold,
        "metrics_on_held_out_test":metrics
    },indent=2))

    out=df.iloc[test][["session_id","user_id","label","attack_type"]].copy()
    out["lstm_autoencoder_score"]=test_scores
    out.to_csv("reports/lstm_autoencoder_scores.csv",index=False)

    print("LSTM Autoencoder — validation-selected threshold, final held-out test")
    print(pd.Series(metrics).to_string())

if __name__=="__main__":
    main()
