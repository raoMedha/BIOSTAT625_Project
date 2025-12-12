#!/usr/bin/env python3
"""
PTB-XL 1D CNN classifier (final pipeline)

- Loads PTB-XL metadata
- Maps SCP codes to diagnostic superclasses (NORM, MI, STTC, CD, HYP)
- Builds a balanced subset of the data
- Preprocesses ECGs: bandpass filter + z-score normalization + downsampling
- Trains a 1D CNN with class weights and early stopping
- Evaluates on train/val/test sets with classification report & confusion matrix
"""

import os
import ast
import sys
import numpy as np
import pandas as pd

import wfdb
from scipy.signal import butter, filtfilt

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


#######################################################################
# Setting up paths and specific data path names #######################
#######################################################################

DATA_DIR = os.path.join("data", "ptbxl")
DB_CSV_PATH = os.path.join(DATA_DIR, "ptbxl_database.csv")
SCP_CSV_PATH = os.path.join(DATA_DIR, "scp_statements.csv")

# Diagnostic superclass to label
LABEL_MAP = {
    "NORM": 0,
    "MI": 1,
    "STTC": 2,
    "CD": 3,
    "HYP": 4,
}

# How many samples per class to use; want a balanced subset so model doesn't predict everything as normal (normal dominates in the sample)
N_PER_CLASS = 1500 

# Re Hz factor (500 Hz is a lot to process... can do 250 Hz with length 2500)
DOWNSAMPLE_FACTOR = 1

# Training hyperparameters
EPOCHS = 40
BATCH_SIZE = 32
VAL_SIZE = 0.15  # of total
TEST_SIZE = 0.15  # of total
RANDOM_STATE = 42


########################################################################
# Preprocessiong  ######################################################
########################################################################

# normalizing the signal, z score style
def normalize_signal(sig: np.ndarray) -> np.ndarray:
    mean = np.mean(sig, axis=0, keepdims=True)
    std = np.std(sig, axis=0, keepdims=True) + 1e-8
    return (sig - mean) / std


# apply bandpass filter to reduce noise in reading
def bandpass_filter(sig: np.ndarray,
                    low: float = 0.5,
                    high: float = 40.0,
                    fs: float = 500.0) -> np.ndarray:

    nyq = fs / 2.0
    b, a = butter(1, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, sig, axis=0)

# full function for preprocessing tasks defined above
def preprocess(sig: np.ndarray) -> np.ndarray:

    sig = bandpass_filter(sig)
    # sig = normalize_signal(sig)
    if DOWNSAMPLE_FACTOR > 1:
        sig = sig[::DOWNSAMPLE_FACTOR, :]
    return sig.astype(np.float32)


# applying preprocessing to data
def load_ecg_from_row(row: pd.Series) -> np.ndarray:
    path = os.path.join(DATA_DIR, row["filename_hr"])
    sig, meta = wfdb.rdsamp(path)
    sig = preprocess(sig)
    return sig


##############################################################
# Mapping Labels #############################################
##############################################################

# create a full matrix with each code mapped
def build_diag_map(scp_csv_path: str) -> dict:
    scp = pd.read_csv(scp_csv_path)
    scp_diag = scp[scp["diagnostic_class"].notna()]
    diag_map = scp_diag.set_index("Unnamed: 0")["diagnostic_class"].to_dict()
    return diag_map

# from the codes, put them in the superclass
def get_superclass_from_scp_codes(row: pd.Series, diag_map: dict):
    codes = row["scp_codes"].keys()
    classes = [diag_map[c] for c in codes if c in diag_map]
    if len(classes) == 0:
        return None
    # pick the first mapped class (PTB-XL practice for single-label experiments)
    return classes[0]


##############################################################
# Model Building #############################################
##############################################################

# 1D CNN infrastructire
def build_model(input_shape, n_classes: int = 5) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv1D(32, kernel_size=7, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(64, kernel_size=7, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(128, kernel_size=7, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(n_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


##############################################################
# Dataset Manupulations ######################################
##############################################################

# create the sample from the main data, balanced subset 
def make_balanced_subset(df_model: pd.DataFrame) -> pd.DataFrame:
    subsets = []
    for label in sorted(df_model["label"].unique()):
        df_cls = df_model[df_model["label"] == label]
        n_take = min(N_PER_CLASS, len(df_cls))
        subsets.append(df_cls.sample(n=n_take, random_state=RANDOM_STATE))
    balanced = pd.concat(subsets).sample(frac=1.0, random_state=RANDOM_STATE)  # shuffle
    return balanced

# weight each class of identifier for ECG
def compute_class_weights(y_train: np.ndarray) -> dict:
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    return {int(c): float(w) for c, w in zip(classes, weights)}


#####################################################
# Main  #############################################
#####################################################

def main():
    print("Python executable:", sys.executable)

    # load data
    df = pd.read_csv(DB_CSV_PATH)

    # parse scp codes
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    # map scp vodes to diagnostic codes
    diag_map = build_diag_map(SCP_CSV_PATH)

    # condense into diagnositc superclasses
    df["diagnostic_superclass"] = df.apply(
        lambda row: get_superclass_from_scp_codes(row, diag_map),
        axis=1,
    )

    # map superclasses to numeric lables
    df["label"] = df["diagnostic_superclass"].map(LABEL_MAP)

    # only keeping rows that have a superclass (i.e. no missing outcome)
    df_model = df[df["label"].notna()].copy()
    print("df_model shape:", df_model.shape)

    print("Class counts in full df_model:")
    print(df_model["label"].value_counts().sort_index())

    # creating the balanced sample across superclass classifications
    df_bal = make_balanced_subset(df_model)
    print("\nBalanced subset shape:", df_bal.shape)
    print("Class counts in balanced subset:")
    print(df_bal["label"].value_counts().sort_index())

    # loading signals 
    X_list = []
    y_list = []

    for idx, row in df_bal.iterrows():
        sig = load_ecg_from_row(row)
        X_list.append(sig)
        y_list.append(int(row["label"]))

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int64)

    print("\nX shape:", X.shape) 
    print("y shape:", y.shape)

    # TRAIN/ VALIDATION/ TEST split!! 
    # Test 
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Train and Validation
    val_fraction_of_temp = VAL_SIZE / (1.0 - TEST_SIZE)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_fraction_of_temp,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    print("\nTrain shape:", X_train.shape, y_train.shape)
    print("Val shape:  ", X_val.shape, y_val.shape)
    print("Test shape: ", X_test.shape, y_test.shape)

    # class weights
    class_weight = compute_class_weights(y_train)
    print("\nClass weights:", class_weight)

    # build and train model 
    input_shape = X_train.shape[1:] 
    model = build_model(input_shape=input_shape, n_classes=len(LABEL_MAP))

    model.summary()

    es_cb = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    rl_cb = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        verbose=1,
        min_lr=1e-5,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=[es_cb, rl_cb],
        verbose=1,
    )

    # Evaluate model

    def evaluate_split(name, X_split, y_split):
        print(f"\n=== {name} performance ===")
        loss, acc = model.evaluate(X_split, y_split, verbose=0)
        print(f"{name} loss: {loss:.4f}")
        print(f"{name} accuracy: {acc:.4f}")

        y_pred_probs = model.predict(X_split, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)

        print("Classification report:")
        print(classification_report(
            y_split,
            y_pred,
            digits=3,
        ))

        print("Confusion matrix:")
        print(confusion_matrix(y_split, y_pred))

    evaluate_split("Train", X_train, y_train)
    evaluate_split("Validation", X_val, y_val)
    evaluate_split("Test", X_test, y_test)

    # Save validation figures to artifacts folder
    import json
    import matplotlib.pyplot as plt  # no confusion_matrix import here

    ARTIFACTS_DIR = os.path.join("artifacts")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # validation confusion matrix (plot)
    y_val_pred_probs = model.predict(X_val, verbose=0)  # <-- FIXED: model.predict
    y_val_pred = np.argmax(y_val_pred_probs, axis=1)
    cm_val = confusion_matrix(y_val, y_val_pred)        # uses global import

    # confusion matrix
    np.save(os.path.join(ARTIFACTS_DIR, "confusion_matrix_val.npy"), cm_val)
    np.savetxt(
        os.path.join(ARTIFACTS_DIR, "confusion_matrix_val.csv"),
        cm_val,
        delimiter=",",
        fmt="%d",
    )

    # test confusion matrix (plot)
    # predictions from model 
    y_test_pred_probs = model.predict(X_test, verbose=0)
    y_test_pred = np.argmax(y_test_pred_probs, axis=1)

    # confusion matrix
    cm_test = confusion_matrix(y_test, y_test_pred, labels=[0, 1, 2, 3, 4])

    # plot matrix, save in artifacts folder
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_test,
        display_labels=['NORM', 'MI', 'STTC', 'CD', 'HYP']
    )
    disp.plot(xticks_rotation=45, cmap="Blues")
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, "cm_test.png"), dpi=150)
    plt.close()


    # loss curve (train val)
    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, "loss_curve.png"))
    plt.close()

    # metrics of model in json format
    final_metrics = {
        "train_loss": float(history.history["loss"][-1]),
        "train_accuracy": float(history.history["accuracy"][-1]),
        "val_loss": float(history.history["val_loss"][-1]),
        "val_accuracy": float(history.history["val_accuracy"][-1]),
        "history": {
            "loss": [float(x) for x in history.history["loss"]],
            "val_loss": [float(x) for x in history.history["val_loss"]],
            "accuracy": [float(x) for x in history.history["accuracy"]],
            "val_accuracy": [float(x) for x in history.history["val_accuracy"]],
        },
        "confusion_matrix_val": cm_val.tolist(),
        "label_map": LABEL_MAP,
    }

    with open(os.path.join(ARTIFACTS_DIR, "metrics_val.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)



if __name__ == "__main__":
    main()
