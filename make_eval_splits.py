"""Create labeled train.csv/test.csv eval splits for each task.

For every task:
  1. Move all current top-level files into a `raw/` snapshot (once), then
     restore task_description.txt to the top level (the pipeline reads it there).
  2. Split the labeled source file from raw/ into train.csv + test.csv, both
     keeping the target column(s). 15% held out as test.
"""
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

TASKS_DIR = "machine_learning_engineering/tasks"
TEST_SIZE = 0.20
SEED = 42

# task -> (source file (as named in raw/), target col(s), stratify?)
TASKS = {
    "adult-income":              ("adult.csv",                              "income",                                            True),
    "california-housing-prices": ("train.csv",                              "median_house_value",                                False),
    "employee-attrition":        ("WA_Fn-UseC_-HR-Employee-Attrition.csv",  "Attrition",                                         True),
    "credit-card-fraud":         ("creditcard.csv",                        "Class",                                             True),
    "medical-insurance":         ("insurance.csv",                          "charges",                                           False),
    "nomad2018-predict-transparent-conductors": ("train.csv",              ["formation_energy_ev_natom", "bandgap_energy_ev"],  False),
    "pima-diabetes":             ("diabetes.csv",                           "Outcome",                                           True),
    "tabular-playground-series-may-2022": ("train.csv",                     "target",                                            True),
    "titanic":                   ("train.csv",                              "Survived",                                          True),
    "wine-quality-red":          ("winequality-red.csv",                    "quality",                                           False),
}


def snapshot_raw(task_dir):
    raw_dir = os.path.join(task_dir, "raw")
    if os.path.exists(raw_dir):
        return  # already snapshotted
    os.makedirs(raw_dir)
    for name in os.listdir(task_dir):
        if name == "raw":
            continue
        shutil.move(os.path.join(task_dir, name), os.path.join(raw_dir, name))
    # restore task_description.txt at top level (pipeline reads it there)
    desc = os.path.join(raw_dir, "task_description.txt")
    if os.path.exists(desc):
        shutil.copy2(desc, os.path.join(task_dir, "task_description.txt"))


def main():
    for task, (src, target, stratify) in TASKS.items():
        task_dir = os.path.join(TASKS_DIR, task)
        if not os.path.isdir(task_dir):
            print(f"SKIP {task}: dir missing")
            continue
        snapshot_raw(task_dir)

        src_path = os.path.join(task_dir, "raw", src)
        df = pd.read_csv(src_path)
        strat = df[target] if stratify else None
        train_df, test_df = train_test_split(
            df, test_size=TEST_SIZE, random_state=SEED, stratify=strat
        )
        train_df.to_csv(os.path.join(task_dir, "train.csv"), index=False)
        test_df.to_csv(os.path.join(task_dir, "test.csv"), index=False)
        print(f"OK   {task}: {len(df)} -> train {len(train_df)} / test {len(test_df)} "
              f"(target={target}, stratify={stratify})")


if __name__ == "__main__":
    main()
