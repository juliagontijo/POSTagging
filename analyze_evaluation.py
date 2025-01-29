import os
from tqdm import tqdm_notebook as tqdm
import numpy as np
import torch
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

device = 'cuda' if torch.cuda.is_available() else 'cpu'

main_file_path = "data/macmorpho-v3.tgz"
test_filepath = "data/macmorpho-test.txt"
train_filepath = "data/macmorpho-train.txt"


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


from tqdm import tqdm

from sklearn.metrics import classification_report as sklearn_classification_report
"""###Evaluate per class"""
def evaluate_model_per_class(model, tokenizer, X_test, y_test, label_classes):
    model.eval()  # Set model to evaluation mode

    y_pred = []
    y_true = []

    with torch.no_grad():
        for sentence, true_tags in tqdm(zip(X_test, y_test), total=len(X_test), desc="Evaluating", unit="sentence"):
            # Tokenize input
            inputs = tokenizer(sentence, return_tensors="pt", is_split_into_words=True, truncation=True)
            outputs = model(**inputs)

            # Get predictions (logits -> argmax -> predicted labels)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            # Map predictions back to the original tags
            word_ids = inputs.word_ids(batch_index=0)
            predicted_tags = []
            for idx, word_id in enumerate(word_ids):
                if word_id is None:  # Skip special tokens
                    continue
                if word_id != word_ids[idx - 1]:  # Only consider the first subword
                    predicted_tags.append(model.config.id2label[predictions[0, idx].item()])

            # Ensure the prediction length matches the true tag length
            if len(predicted_tags) == len(true_tags):
                y_pred.append(predicted_tags)
                y_true.append(true_tags)
            else:
                print(f"Length mismatch: predicted {len(predicted_tags)}, true {len(true_tags)}")

    # Flatten the lists for classification report
    y_true_flat = [tag for sentence in y_true for tag in sentence]
    y_pred_flat = [tag for sentence in y_pred for tag in sentence]

    # Generate and print the classification report
    report = sklearn_classification_report(y_true_flat, y_pred_flat, labels=label_classes, zero_division=0)
    print(report)

    return report


# Specify the path to the saved model
model_dir = "results"

# Load the saved model and tokenizer
model = AutoModelForTokenClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
print("Model and tokenizer loaded successfully!")

train_data = MacMorphoData(train_filepath, tokenizer, max_length=50)
test_data = MacMorphoData(test_filepath, tokenizer, max_length=50)

# print("\n\n\nTest data\n\n\n")
# print(test_data.wt[0])
# print(test_data.words[1])
# print(test_data.tags[1])


train_dataset = POSDataset(train_data.processed_data)
eval_dataset = POSDataset(test_data.processed_data)

# print("\n\n\nEval dataset\n\n\n")
# print(eval_dataset[0])

print("Train and Test DATASET ready\n\n")

X_test = [[word for word, tag in sentence] for sentence in test_data.wt]
y_test = [[tag for word, tag in sentence] for sentence in test_data.wt]

print(X_test[0])
print(y_test[0])


label_classes = test_data.unique_tags
print(test_data.unique_tags)


# Call the evaluation function
metrics_per_class = evaluate_model_per_class(model, tokenizer, X_test, y_test, label_classes)

import pickle
per_class_pickle_file = "results/evaluation_metrics_per_class.pkl"
with open(per_class_pickle_file, "wb") as f:
    pickle.dump(metrics_per_class, f)
print(f"\n\nEvaluation saved to {per_class_pickle_file}\n\n")

