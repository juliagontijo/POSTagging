
import os

main_file_path = "data/macmorpho-v3.tgz"
test_filepath = "data/macmorpho-test.txt"
train_filepath = "data/macmorpho-train.txt"


import os
from tqdm import tqdm_notebook as tqdm
import numpy as np
import torch
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""##Get Train Data"""

class MacMorphoData:
    def __init__(self, filepath, tokenizer, max_length=50):
        """
        filepath: path to the MacMorpho file (e.g., '/content/macmorpho-train.txt')
        tokenizer: a Hugging Face tokenizer (e.g., AutoTokenizer.from_pretrained(...))
        max_length: maximum sequence length for tokenization
        """
        self.filepath = filepath
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Read lines
        with open(self.filepath, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')

        # Each line -> list of "word_POS" strings
        items = [line.strip().split() for line in lines if line.strip()]

        # Turn each sentence into a list of (word, tag) tuples
        self.wt = [
            [tuple(unit.split('_')) for unit in sentence]
            for sentence in items
        ]

        # Flatten all (word, tag) pairs across the entire file
        pairs = [(word, tag) for sentence in self.wt for (word, tag) in sentence]

        # Separate words and tags
        self.words, self.tags = zip(*pairs)  # 'words' and 'tags' are tuples now
        self.words = list(self.words)
        self.tags = list(self.tags)

        # Build unique tags
        self.unique_tags = sorted(list(set(self.tags)))

        # Tag <-> index maps
        self.tag2idx = {tag: idx for idx, tag in enumerate(self.unique_tags)}
        self.idx2tag = {idx: tag for tag, idx in self.tag2idx.items()}

        # Now let's preprocess (tokenize) all sentences
        self.processed_data = self._process_all_sentences()
        # self.processed_data is a list of dicts:
        #   [{'input_ids':..., 'attention_mask':..., 'labels':...}, ...]

    def _process_sentence(self, sentence):
        """
        sentence: list of (word, tag) pairs, e.g. [("Jersei","N"), ("atinge","V"), ("média","N")]
        returns a dict with input_ids, attention_mask, labels
        """
        words = [w for (w, t) in sentence]
        tags = [t for (w, t) in sentence]

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            padding='max_length',    # or True, or 'longest'
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'      # return PyTorch tensors
        )

        input_ids = encoding['input_ids'][0]        # shape (seq_len,)
        attention_mask = encoding['attention_mask'][0]  # shape (seq_len,)

        # Use word_ids() to map back to original words
        word_ids = encoding.word_ids(batch_index=0)

        # Build label_ids aligned with subwords
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                # Special tokens [CLS], [SEP], or padded subwords
                label_ids.append(-100)
            else:
                # Use the word_id to find the original tag
                original_tag = tags[word_id]
                label_id = self.tag2idx[original_tag]
                label_ids.append(label_id)

        # Return a dictionary
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label_ids)
        }

    def _process_all_sentences(self):
        """
        Helper to process/tokenize all sentences in self.wt
        """
        processed = []
        for sentence in self.wt:
            processed_example = self._process_sentence(sentence)
            processed.append(processed_example)
        return processed

    def __len__(self):
        """
        Number of sentences (before tokenization).
        """
        return len(self.wt)

import torch
from torch.utils.data import Dataset

class POSDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings  # list of dicts

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return {k: v for k, v in self.encodings[idx].items()}

"""##Model"""



tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

train_data = MacMorphoData(train_filepath, tokenizer, max_length=50)
test_data = MacMorphoData(test_filepath, tokenizer, max_length=50)


print("Train and Test data ready\n\n")

from transformers import DataCollatorForTokenClassification
from torch.utils.data import DataLoader

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

train_dataset = POSDataset(train_data.processed_data)
eval_dataset = POSDataset(test_data.processed_data)


training_args = TrainingArguments(
    output_dir="results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    learning_rate=5e-5
)

def align_predictions(predictions, label_ids):
    preds = predictions.argmax(-1)
    batch_size, seq_len = preds.shape
    out_preds, out_labels = [], []

    for i in range(batch_size):
        pred_i = []
        label_i = []
        for j in range(seq_len):
            if label_ids[i][j] != -100:
                pred_i.append(model.config.id2label[preds[i][j].item()])
                label_i.append(model.config.id2label[label_ids[i][j].item()])
        out_preds.append(pred_i)
        out_labels.append(label_i)
    return out_preds, out_labels

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.from_numpy(logits).argmax(-1)  # Get predicted labels
    label_ids = torch.from_numpy(labels)

    pred_tags, true_tags = align_predictions(predictions, label_ids)

    # Compute overall metrics
    precision = precision_score(true_tags, pred_tags)
    recall = recall_score(true_tags, pred_tags)
    f1 = f1_score(true_tags, pred_tags)

    # Compute class-wise metrics using seqeval
    class_report = classification_report(true_tags, pred_tags, output_dict=True)

    # Return all metrics, including the class report
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "class_report": class_report  # Add the class report here
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

####################
# 6) Evaluate
####################
print("Starting evaluation\n\n")
metrics = trainer.evaluate()
print("Evaluation ended\n\n")
print("Evaluation:", metrics)

import pickle

pickle_file = "results/evaluation_metrics.pkl"
with open(pickle_file, "wb") as f:
    pickle.dump(metrics, f)
print(f"\n\nEvaluation saved to {pickle_file}\n\n")


class_report = metrics.get("class_report", None)
if class_report:
    # Extract and display precision for each class
    for class_name, class_metrics in class_report.items():
        if class_name not in ["accuracy"]:  # Skip overall accuracy
            print(f"{class_name}: Precision = {class_metrics['precision']:.2f}")

    # Identify highest/lowest precision classes
    precision_per_class = {
        class_name: class_metrics["precision"]
        for class_name, class_metrics in class_report.items()
        if class_name not in ["accuracy"]
    }

    # Sort by precision
    sorted_classes = sorted(precision_per_class.items(), key=lambda x: x[1], reverse=True)

    # Display results
    print("\nClasses with highest precision:")
    for class_name, precision in sorted_classes[:3]:  # Top 3 classes
        print(f"{class_name}: {precision:.2f}")

    print("\nClasses with lowest precision:")
    for class_name, precision in sorted_classes[-3:]:  # Bottom 3 classes
        print(f"{class_name}: {precision:.2f}")

    print("\nPrecision for all classes (sorted):")
    for class_name, precision in sorted_classes:
        print(f"{class_name}: {precision:.2f}")
else:
    print("Class report is not available in the metrics.")