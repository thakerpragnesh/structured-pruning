"""GLUE data, metrics, and the Trainer factory for the pruning study.

Generalises `transformer_pruning/experiments/_mrpc.py` (MRPC only) to the
GLUE tasks, keeping its two load-bearing rules:

1. Every variant starts from one shared fine-tuned baseline, so the only
   difference between rows is the pruning criterion.
2. A *fresh* Trainer is built after every surgery (`new_trainer`). A Trainer
   caches its optimizer around the parameter objects that existed at its
   first `train()`; surgery replaces those modules, so a reused Trainer
   would "heal" detached pre-surgery parameters and leave the pruned model
   untrained.

Metrics are computed with scikit-learn/scipy (GLUE's official definitions),
not the `evaluate` package, so no metric script is downloaded at run time.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class TaskSpec:
    name: str
    text_keys: tuple            # one or two text columns
    num_labels: int             # 1 = regression (STS-B)
    primary_metric: str         # what the summary tables rank by
    validation_split: str = "validation"


# Primary metric = what summary tables rank by; every metric is still recorded.
# MRPC/QQP use accuracy, not F1: both are majority-positive, so an
# always-"yes" model already scores ~0.81 F1 on MRPC, which hides the
# differences between pruning criteria the study is trying to measure.
TASKS = {
    "mrpc": TaskSpec("mrpc", ("sentence1", "sentence2"), 2, "accuracy"),
    "rte": TaskSpec("rte", ("sentence1", "sentence2"), 2, "accuracy"),
    "sst2": TaskSpec("sst2", ("sentence",), 2, "accuracy"),
    "cola": TaskSpec("cola", ("sentence",), 2, "matthews"),
    "qnli": TaskSpec("qnli", ("question", "sentence"), 2, "accuracy"),
    "stsb": TaskSpec("stsb", ("sentence1", "sentence2"), 1, "pearson"),
    "qqp": TaskSpec("qqp", ("question1", "question2"), 2, "accuracy"),
    "mnli": TaskSpec("mnli", ("premise", "hypothesis"), 3, "accuracy", "validation_matched"),
}


def set_seeds(seed: int) -> None:
    """Seed every RNG the study touches: the random-saliency baseline, data
    shuffling during healing, and dropout all depend on it."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics_for(task: TaskSpec):
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

    def compute(eval_preds):
        logits, labels = eval_preds
        if task.num_labels == 1:
            preds = np.squeeze(logits)
            return {"pearson": float(pearsonr(preds, labels)[0]),
                    "spearman": float(spearmanr(preds, labels)[0])}
        preds = np.argmax(logits, axis=-1)
        out = {"accuracy": float(accuracy_score(labels, preds))}
        if task.num_labels == 2:
            out["f1"] = float(f1_score(labels, preds))
            out["matthews"] = float(matthews_corrcoef(labels, preds))
        return out

    return compute


class ListDataset(torch.utils.data.Dataset):
    """A list of feature dicts as a torch Dataset (used by `--smoke`)."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


class GlueHarness:
    """Dataset, tokenizer, metric and Trainer factory for one GLUE task."""

    def __init__(self, task: str, model_name: str, work_dir, seed: int, *, epochs: float = 3,
                 batch_size: int = 32, learning_rate: float = 2e-5, max_length: int = 128,
                 train_limit: int | None = None, eval_limit: int | None = None, smoke: bool = False):
        if task not in TASKS:
            raise ValueError(f"unknown task {task!r}, expected one of {sorted(TASKS)}")
        self.task = TASKS[task]
        self.model_name = model_name
        self.work_dir = work_dir
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.smoke = smoke
        if smoke:
            self._build_synthetic()
        else:
            self._build_glue(max_length, train_limit, eval_limit)

    # ----------------------------------------------------------------- data
    def _build_glue(self, max_length, train_limit, eval_limit):
        from datasets import load_dataset
        from transformers import AutoTokenizer, DataCollatorWithPadding

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        raw = load_dataset("nyu-mll/glue", self.task.name)
        keys = self.task.text_keys

        def tokenize(batch):
            return self.tokenizer(*(batch[k] for k in keys), truncation=True, max_length=max_length)

        train = raw["train"].shuffle(seed=self.seed)
        if train_limit:
            train = train.select(range(min(train_limit, len(train))))
        val = raw[self.task.validation_split]
        if eval_limit:
            val = val.select(range(min(eval_limit, len(val))))
        keep = {"input_ids", "attention_mask", "token_type_ids", "label"}
        self.train_set = train.map(tokenize, batched=True)
        self.train_set = self.train_set.remove_columns([c for c in self.train_set.column_names if c not in keep])
        self.eval_set = val.map(tokenize, batched=True)
        self.eval_set = self.eval_set.remove_columns([c for c in self.eval_set.column_names if c not in keep])
        self.collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def _build_synthetic(self):
        """Random tokens, real padding, learnable label rule -- exercises the
        whole Trainer / calibration / surgery path with no downloads."""
        from transformers import default_data_collator

        g = torch.Generator().manual_seed(self.seed)

        def rows(n):
            out = []
            for _ in range(n):
                length = int(torch.randint(8, 17, (1,), generator=g))
                ids = torch.randint(5, 200, (16,), generator=g)
                mask = torch.zeros(16, dtype=torch.long)
                mask[:length] = 1
                ids = ids * mask
                # Label = which half of the vocab the first token is in: the
                # pooled [CLS] position sees it directly, so a tiny model learns
                # it in a few epochs and pruning visibly moves the metric.
                label = int(ids[0] >= 100)
                out.append({"input_ids": ids, "attention_mask": mask,
                            "token_type_ids": torch.zeros(16, dtype=torch.long), "labels": label})
            return out

        self.tokenizer = None
        self.train_set = ListDataset(rows(256))
        self.eval_set = ListDataset(rows(96))
        self.collator = default_data_collator

    def calibration_batches(self, limit: int = 512, batch_size: int = 32, with_labels: bool = False):
        """Collated training batches for the calibrators -- same tokenization
        and collator as evaluation, so the statistics describe the inputs the
        model is actually scored on. `with_labels=True` keeps `labels`, which
        gradient head importance needs for the task loss."""
        from torch.utils.data import DataLoader, Subset

        ds = Subset(self.train_set, range(min(limit, len(self.train_set))))
        batches = []
        for batch in DataLoader(ds, batch_size=batch_size, collate_fn=self.collator):
            batch = dict(batch)
            if "label" in batch:
                batch["labels"] = batch.pop("label")
            if not with_labels:
                batch.pop("labels", None)
            batches.append(batch)
        return batches

    # --------------------------------------------------------------- model
    def new_model(self):
        """The pretrained checkpoint with a fresh task head (or, in smoke
        mode, a tiny random-init BERT)."""
        from transformers import AutoModelForSequenceClassification, BertConfig, BertForSequenceClassification

        if self.smoke:
            config = BertConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=8,
                                intermediate_size=128, vocab_size=256, num_labels=self.task.num_labels)
            return BertForSequenceClassification(config)
        return AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.task.num_labels)

    def new_trainer(self, model, epochs: float | None = None, progress: bool = False,
                    train_examples: int | None = None):
        """A *fresh* Trainer around `model` -- call again after every surgery
        (see the module docstring). `progress=True` shows a progress bar and
        loss logs (used for the long baseline fine-tune only)."""
        from transformers import PrinterCallback, Trainer, TrainingArguments

        args = TrainingArguments(
            output_dir=str(self.work_dir / "trainer_tmp"),
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size * 2,
            num_train_epochs=self.epochs if epochs is None else epochs,
            weight_decay=0.01,
            warmup_ratio=0.1,
            eval_strategy="no",
            save_strategy="no",
            logging_strategy="steps",
            logging_steps=200,
            fp16=torch.cuda.is_available(),
            seed=self.seed,
            data_seed=self.seed,
            report_to=[],
            disable_tqdm=not progress,
            dataloader_num_workers=2 if torch.cuda.is_available() else 0,
        )
        train_set = self.train_set
        if train_examples is not None and train_examples < len(train_set):
            # The training set is already shuffled by seed, so a prefix is a
            # seed-dependent random subset, identical across variants.
            train_set = torch.utils.data.Subset(train_set, range(train_examples))
        trainer = Trainer(model=model, args=args, train_dataset=train_set,
                          eval_dataset=self.eval_set, data_collator=self.collator,
                          compute_metrics=compute_metrics_for(self.task))
        trainer.remove_callback(PrinterCallback)  # run_study prints its own one-line summaries
        return trainer

    def evaluate(self, model) -> dict:
        """Validation metrics with the `eval_` prefix stripped."""
        metrics = self.new_trainer(model).evaluate()
        return {k[len("eval_"):]: v for k, v in metrics.items()
                if k.startswith("eval_") and isinstance(v, (int, float))}

    def train(self, model, epochs: float | None = None, progress: bool = False,
              train_examples: int | None = None) -> None:
        if (self.epochs if epochs is None else epochs) > 0:
            self.new_trainer(model, epochs=epochs, progress=progress, train_examples=train_examples).train()
