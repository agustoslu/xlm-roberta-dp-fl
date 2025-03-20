import os
from pathlib import Path
from datasets import load_dataset
import pandas as pd 

home_dir = Path(__file__).parent
DATA_PATH = home_dir / "dataset"
DATA_PATH.mkdir(parents=True, exist_ok=True)

xnli_dataset = load_dataset("facebook/xnli", "all_languages")
for split in xnli_dataset.keys():
    xnli_dataset[split].to_pandas().to_csv(DATA_PATH / f"{split}.csv", index=False)

train_path = os.path.join(DATA_PATH, "train.csv")
dev_path = os.path.join(DATA_PATH, "validation.csv")
test_path = os.path.join(DATA_PATH, "test.csv")