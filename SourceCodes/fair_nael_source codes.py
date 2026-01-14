import os
import json
import csv
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import datetime


# Hybrid Model for NAEL Project to decode the ALzheimer's disease from CJD and CNTRL.

class CNNTransformerEEG(nn.Module):
    def __init__(self, n_channels=19, n_times=1280, n_classes=3,
                 cnn_out_channels=64, n_heads=2, n_layers=2):
        super().__init__()

        # ---- Simple CNN feature extractor (NO padding) ----
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=n_channels,
                out_channels=cnn_out_channels,
                kernel_size=25,
                padding=0
            ),
            nn.BatchNorm1d(cnn_out_channels),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=4),

            nn.Conv1d(
                in_channels=cnn_out_channels,
                out_channels=cnn_out_channels,
                kernel_size=15,
                padding=0
            ),
            nn.BatchNorm1d(cnn_out_channels),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=4),
        )

        # ---- Transformer encoder ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cnn_out_channels,
            nhead=n_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # ---- Classifier ----
        self.classifier = nn.Linear(cnn_out_channels, n_classes)

    def forward(self, x):
        # x: (B, 19, 1280)

        feats = self.cnn(x)              # (B, C, T')
        feats = feats.permute(0, 2, 1)   # (B, T', C)
 
        feats = self.transformer(feats)  # (B, T', C)

        pooled = feats.mean(dim=1)       # (B, C)
        out = self.classifier(pooled)    # (B, 3)
        return out



### LOAD AND CLEAN DATA AND SAVE IT TO FOLLOWING FORMAT.
# 
#     Load cleaned .npz EEG files, split into 5-second epochs.

#     CLEANED AND PROCESSED DATA SHOULD BE: 
#         data_list: list of np.arrays (n_epochs, n_channels, epoch_samples)
#         labels_list: list of np.arrays (n_epochs,)
#         subject_ids: list of subject identifiers



### LOSO split
def loso_split(data_list, labels_list, subject_ids):
    """
    Generator for Leave-One-Subject-Out (LOSO) splits.
    Yields (X_train, y_train, X_test, y_test, subject_id)
    """
    n_subjects = len(subject_ids)
    for i in range(n_subjects):
        X_test = data_list[i]
        y_test = labels_list[i]
        X_train = np.concatenate([data_list[j] for j in range(n_subjects) if j != i], axis=0)
        y_train = np.concatenate([labels_list[j] for j in range(n_subjects) if j != i], axis=0)
        print("subject in the training and testing", )
        yield X_train, y_train, X_test, y_test, subject_ids[i]



def save_onnx_model(model, n_channels, n_times, filename):
    model.eval()
    dummy_input = torch.randn(1, n_channels, n_times)
    torch.onnx.export(
        model,
        dummy_input,
        filename,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input":{0:"batch"}, "output":{0:"batch"}},
    )
    print(f"Saved ONNX model: {filename}")


# ------- Small dataset wrapper for DataLoader ----------
class ADDatasetTorch(Dataset):
    """
    Torch Dataset wrapper for (n_epochs, n_channels, n_times) arrays.
    Returns (x, y) where x: float32 (n_channels, n_times)
    """
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        y = int(self.y[idx])
        # models can accept (B, C, T) or (B, 1, C, T) - handled by model forward
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def loso_train_and_eval(
    data_list, labels_list, subject_ids,
    make_model_fn,
    out_dir,
    batch_size,
    n_epochs,
    lr,
    weight_decay,
    device,
    k_folds_train
):
    import os, json, csv, time, datetime
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        classification_report,
        balanced_accuracy_score,
        precision_recall_fscore_support,
        roc_auc_score
    )
    from sklearn.preprocessing import label_binarize

    per_subject_metrics = []

    # ---------- Global aggregators ----------
    all_confusion_matrices = []
    all_subject_accs = []
    all_subject_bal_accs = []
    all_subject_f1s = []
    all_subject_prec = []
    all_subject_recall = []
    all_subject_auc = []
    all_inference_times = []

    # Create global label list for confusion matrix ordering
    label_set = sorted(list({lab for labs in labels_list for lab in labs.tolist()})) \
        if isinstance(labels_list[0], np.ndarray) else [0, 1, 2]
    labels_for_cm = label_set

    for X_train_all, y_train_all, X_test, y_test, subject_id in loso_split(
            data_list, labels_list, subject_ids):

        print("\n==========================")
        print("Held-out test subject:", subject_id)
        print("==========================")

        # ensure numpy arrays
        X_train_all = np.asarray(X_train_all)
        y_train_all = np.asarray(y_train_all)
        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test)

        n_channels = X_train_all.shape[1]
        n_times = X_train_all.shape[2]
        print(f"Total training samples {len(X_train_all)} and test samples {len(X_test)}")

        # ---------- K-Fold on training set ----------
        skf = StratifiedKFold(
            n_splits=k_folds_train,
            shuffle=True,
            random_state=RANDOM_SEED
        )

        best_val_acc_overall = -np.inf
        best_model_state_overall = None

        early_stop_patience = 5
        early_stop_min_delta = 1e-4

        for fold_idx, (tr_idx, val_idx) in enumerate(
                skf.split(X_train_all, y_train_all), start=1):

            print(f"\n--- Train CV fold {fold_idx}/{k_folds_train} ---")

            X_tr, y_tr = X_train_all[tr_idx], y_train_all[tr_idx]
            X_val, y_val = X_train_all[val_idx], y_train_all[val_idx]

            train_loader = DataLoader(
                ADDatasetTorch(X_tr, y_tr),
                batch_size=batch_size,
                shuffle=True
            )
            val_loader = DataLoader(
                ADDatasetTorch(X_val, y_val),
                batch_size=batch_size,
                shuffle=False
            )

            # instantiate model fresh
            model = make_model_fn.to(device)

            opt = torch.optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
            criterion = nn.CrossEntropyLoss()

            best_val_acc_fold = -np.inf
            best_state_fold = None
            epochs_no_improve = 0

            for epoch in range(1, n_epochs + 1):
                # ---- Training ----
                model.train()
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    opt.step()

                # ---- Validation ----
                model.eval()
                preds, gts = [], []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(device)
                        out = model(xb)
                        preds.append(out.argmax(1).cpu().numpy())
                        gts.append(yb.numpy())

                preds = np.concatenate(preds)
                gts = np.concatenate(gts)
                val_acc = accuracy_score(gts, preds)

                print(f"[Fold {fold_idx} | Epoch {epoch}] Val Acc: {val_acc:.4f}")

                if val_acc > best_val_acc_fold + early_stop_min_delta:
                    best_val_acc_fold = val_acc
                    best_state_fold = {
                        k: v.cpu().clone()
                        for k, v in model.state_dict().items()
                    }
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= early_stop_patience:
                    print(f"[Fold {fold_idx}] Early stopping triggered")
                    break

            if best_val_acc_fold > best_val_acc_overall:
                best_val_acc_overall = best_val_acc_fold
                best_model_state_overall = best_state_fold

            del model, opt
            torch.cuda.empty_cache()

        # ---------- Final model for held-out subject ----------
        model_final = CNNTransformerEEG(
            n_channels=n_channels,
            n_times=n_times,
            n_classes=N_CLASSES
        ).to(device)

        if best_model_state_overall is not None:
            model_final.load_state_dict(best_model_state_overall)

        model_final.eval()

        # ---------- Save per-subject directory ----------
        subj_dir = os.path.join(out_dir, f"subject_{subject_id}")
        os.makedirs(subj_dir, exist_ok=True)

        # ---- TorchScript (for Edge-AI) ----
        script_path = os.path.join(subj_dir, f"scripted_best_model_{subject_id}.pt")
        scripted_model = torch.jit.script(model_final)
        scripted_model.save(script_path)
        print("Saved scripted model:", script_path)

        # ---- ONNX export (optional Edge runtime) ----
        try:
            dummy = torch.randn(1, n_channels, n_times).to(device)
            onnx_path = os.path.join(subj_dir, f"best_model_subject_{subject_id}.onnx")
            torch.onnx.export(
                model_final,
                dummy,
                onnx_path,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                opset_version=11
            )
            print("Saved ONNX model:", onnx_path)
        except Exception as e:
            print("ONNX export failed:", e)

        # ---------- Evaluate on held-out subject ----------
        test_loader = DataLoader(
            ADDatasetTorch(X_test, y_test),
            batch_size=batch_size,
            shuffle=False
        )

        preds, gts, probs_list = [], [], []

        start_time = time.time()
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                out = model_final(xb)
                prob = torch.softmax(out, dim=1).cpu().numpy()  # probabilities
                pred = out.argmax(1).cpu().numpy()

                preds.append(pred)
                gts.append(yb.numpy())
                probs_list.append(prob)

        inference_time = time.time() - start_time

        preds = np.concatenate(preds)
        gts = np.concatenate(gts)
        probs = np.concatenate(probs_list)

        acc = accuracy_score(gts, preds)
        bal_acc = balanced_accuracy_score(gts, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            gts, preds, average="macro", zero_division=0
        )
        cm = confusion_matrix(gts, preds, labels=labels_for_cm)
        avg_inf_time = inference_time / len(gts)

        # ---------- Compute per-subject ROC-AUC ----------
        y_true_bin = label_binarize(gts, classes=labels_for_cm)
        try:
            subject_auc = roc_auc_score(y_true_bin, probs, average="macro", multi_class="ovr")
        except ValueError:
            subject_auc = float("nan")
        print(f"Subject {subject_id} macro ROC-AUC: {subject_auc:.4f}")

        # ---- Save test input & labels for Edge-AI testing ----
        np.save(os.path.join(subj_dir, f"test_input_subject_{subject_id}.npy"),
                X_test.astype(np.float32))
        np.save(os.path.join(subj_dir, f"test_labels_subject_{subject_id}.npy"),
                y_test.astype(np.int64))

        # ---------- Collect metrics ----------
        per_subject_metrics.append({
            "subject_id": subject_id,
            "n_test_epochs": int(len(gts)),
            "test_acc": acc,
            "balanced_acc": bal_acc,
            "macro_f1": f1,
            "precision": precision,
            "recall": recall,
            "roc_auc": subject_auc,
            "confusion_matrix": cm.tolist(),
            "inference_time_sec": inference_time,
            "avg_time_per_epoch_sec": avg_inf_time
        })

        all_confusion_matrices.append(cm)
        all_subject_accs.append(acc)
        all_subject_bal_accs.append(bal_acc)
        all_subject_f1s.append(f1)
        all_subject_prec.append(precision)
        all_subject_recall.append(recall)
        all_subject_auc.append(subject_auc)
        all_inference_times.append(avg_inf_time)

        del model_final
        torch.cuda.empty_cache()

    # ---------- Aggregated results ----------
    agg_cm = np.sum(np.stack(all_confusion_matrices), axis=0)
    mean_acc = float(np.mean(all_subject_accs))
    std_acc = float(np.std(all_subject_accs))
    mean_prec = float(np.mean(all_subject_prec))
    std_prec = float(np.std(all_subject_prec))
    mean_recall = float(np.mean(all_subject_recall))
    std_recall = float(np.std(all_subject_recall))
    ci95 = float(1.96 * std_acc / np.sqrt(len(all_subject_accs)))

    results = {
        "per_subject": per_subject_metrics,
        "aggregate": {
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "mean_prec": mean_prec,
            "std_prec": std_prec,
            "mean_recall": mean_recall,
            "std_recall": std_recall,
            "ci95_accuracy": ci95,
            "mean_balanced_accuracy": float(np.mean(all_subject_bal_accs)),
            "mean_macro_f1": float(np.mean(all_subject_f1s)),
            "mean_macro_roc_auc": float(np.nanmean(all_subject_auc)),
            "aggregated_confusion_matrix": agg_cm.tolist(),
            "mean_inference_time_sec": float(np.mean(all_inference_times))
        }
    }

    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---- Save JSON ----
    json_path = os.path.join(out_dir, f"loso_results_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved JSON results:", json_path)

    # ---- Save CSV summary ----
    csv_path = os.path.join(out_dir, f"loso_summary_{timestamp}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "subject_id",
            "n_test_epochs",
            "test_acc",
            "balanced_acc",
            "macro_f1",
            "precision",
            "recall",
            "roc_auc",
            "avg_time_per_epoch_sec"
        ])
        for r in per_subject_metrics:
            writer.writerow([
                r["subject_id"],
                r["n_test_epochs"],
                r["test_acc"],
                r["balanced_acc"],
                r["macro_f1"],
                r["precision"],
                r["recall"],
                r["roc_auc"],
                r["avg_time_per_epoch_sec"]
            ])
    print("Saved CSV summary:", csv_path)

    return per_subject_metrics



# ---------- To Run the models ----------
if __name__ == "__main__":

    ROOT_DIR = "/home/.../EEG_AD_CUT/working/cleaned_filtered/"   # folder with AD/, CJD/, CNTRL/ subfolders containing .npz
    
    MODEL_NAME = "hybrid"              
    OUT_DIR = "hybrid_loso_results"
    BATCH_SIZE = 64
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    N_EPOCHS = 30
    K_FOLDS_TRAIN = 5                    # k-fold CV on training set to choose best fold model
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SFREQ = 256
    EPOCH_LEN_S = 5
    N_CLASSES = 3                         # AD, CJD, CNTRL
    RANDOM_SEED = 2025
    # -----------------------------------

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    os.makedirs(OUT_DIR, exist_ok=True)
    # load data using your provided loader

    #data_list, labels_list, subject_ids = YOUR DATA CLEANING FUNCTION TO CALL 
    # Sanity check
    print("Loaded subjects:", subject_ids)
    print([d.shape for d in data_list])

     # CNN + DNN WITH TRANSFORMEER
    model = CNNTransformerEEG(n_channels=19, n_times=1280, n_classes=3) # CALL PROPOSED HYBRID MODEL
   

    # tHIS WILL CREATE AND SAVE THE LOSO RESULTS AT YOUR CREATED DIRECTORY
    results = loso_train_and_eval(data_list, labels_list, subject_ids, model,
                                  out_dir=OUT_DIR, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS,
                                  lr=LR, weight_decay=WEIGHT_DECAY, device=DEVICE, k_folds_train=K_FOLDS_TRAIN)

    print("LOSO finished. Summary per-subject saved to disk.")

