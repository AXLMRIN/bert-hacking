import pandas as pd 
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from torch import no_grad
import numpy as np
from sklearn.metrics import classification_report

import toolbox as tb

FREEZE_FIRST_N_LAYERS = 4
THRESHOLD_TO_TRY = [0.6, 0.75, 0.8]

loop_config = tb.LoopConfig(
    dataset_name="TestEtienne-AD_POPULUM",
    dichotomization_label="ad_populum", 
    
    model_name = 'camembert/camembert-base', 
    n_epochs = 3, 
    learning_rate = 0.00001, 
    weight_decay = 0.01,
    batch_size = 16,

    device_batch_size_for_prediction = 64
)


df = pd.read_csv("./data/annotations_ldp-radjour-phr-testbreak2_all_train.csv", usecols=["id_phr", "AdPop_restr_EO_dataset", "AdPop_restr_EO_labels", "AdPop_restr_EO_text"])
df = df.rename(columns={
    "id_phr": "ID",
    "AdPop_restr_EO_dataset":"split",
    "AdPop_restr_EO_labels":"LABEL",
    "AdPop_restr_EO_text":"TEXT",
})
df = df[["ID", "split", "LABEL", "TEXT"]]

df = df.loc[df["LABEL"].notna()]
df["LABEL"] = df["LABEL"].map({"AD_POPULUM":'ad_populum', 'NOT':'not-ad_populum'})
LABEL2ID = {"ad_populum": 1, "not-ad_populum":0}
ID2LABEL = {1: "ad_populum", 0:"not-ad_populum"}

print(df.groupby(['split','LABEL']).size())


dsd_loop = DatasetDict({
    "train": Dataset.from_pandas(df.loc[df.split == "train"]),
    "test": Dataset.from_pandas(df.loc[df.split == "valid"])
})
temp = dsd_loop["train"].train_test_split(0.2)
dsd_loop["train"] = temp["train"]
dsd_loop["eval"] = temp["test"]

# Tokenize
TOKENIZER = AutoTokenizer.from_pretrained(loop_config.model_name)

def tokenization(row:dict):
    tokenized_entry = TOKENIZER(
        row["TEXT"], 
        truncation = True, 
        padding = "max_length",
        max_length = 300
    )
    return {
        **row.copy(), 
        "labels": int(LABEL2ID[row["LABEL"]]),
        **tokenized_entry
    }

dsd_loop = dsd_loop.map(tokenization)
print(dsd_loop)

model = AutoModelForSequenceClassification.from_pretrained(
    loop_config.model_name,
    num_labels = len(LABEL2ID),
    id2label   = ID2LABEL,
    label2id   = LABEL2ID,
)

for name, param in model.base_model.named_parameters():
    if np.array([
        name.startswith(f'encoder.layer.{i}') 
        for i in range(FREEZE_FIRST_N_LAYERS)
    ]).any():
        param.requires_grad = False

training_args = tb.load_training_arguments(loop_config)
best_model_checkpoint, trainer_logs = tb.train_model(model, training_args,dsd_loop,loop_config)

model = AutoModelForSequenceClassification.from_pretrained(best_model_checkpoint)
device = tb.get_device()
print(f"Predict on {device}")


ds = dsd_loop["test"].with_format("torch", device=device)

if loop_config.test_mode: ds = ds.select(range(20))

model = model.to(device=device)
model.eval()
if str(device)=="cuda": model = model.bfloat16()

ID_= []
GS_ = []
PRED_ = []
PROBS_POS = [] 
for batch in tqdm(ds.batch(loop_config.device_batch_size_for_prediction), desc="Prediction"):
    with no_grad():
        probs = (
            model(input_ids = batch["input_ids"], attention_mask= batch["attention_mask"])
            .logits.cpu().softmax(1).float().numpy()
        )
    PROBS_POS += [probs[:,1]]
    y_pred = np.argmax(probs, axis = 1).reshape(-1)
    
    ID_ += batch["ID"]
    GS_ += batch["LABEL"]
    PRED_ += [ID2LABEL[int(y)] for y in y_pred]

PROBS_POS = np.concatenate(PROBS_POS)

## Evaluate metrics 
print("#"*50)
print("Regular: argmax")
print(classification_report(y_true = GS_, y_pred = PRED_, zero_division=np.nan))

for t in THRESHOLD_TO_TRY:
    print("#"*50)
    print(f"Threshold {t}")
    print(classification_report(y_true=GS_, y_pred = [ID2LABEL[int(p > t)] for p in PROBS_POS], zero_division=np.nan))