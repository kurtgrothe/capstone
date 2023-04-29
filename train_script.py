import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

class TextClassifier:
    def __init__(self, train_data, test_data, num_labels, tokenizer='distilbert-base-uncased'):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer)
        self.train_dataloader = self.create_data_loader(train_data.values, self.tokenizer)
        self.test_dataloader = self.create_data_loader(test_data.values, self.tokenizer)
        self.num_labels = num_labels
        
    def create_data_loader(self, data, tokenizer, max_length=512, num_workers=8, pin_memory=True):
        def collate_fn(batch):
            features = [item[:-1] for item in batch]
            labels = [item[-1] for item in batch]

            feature_strings = [' '.join([str(f) for f in feature]) for feature in features]
            inputs = tokenizer(feature_strings, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)

            return {
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask'],
                'labels': torch.tensor(labels, dtype=torch.long)
            }

        return DataLoader(data, batch_size=32, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    
    @staticmethod
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        acc = accuracy_score(labels, predictions)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train_from_scratch(self, output_dir, num_train_epochs=3):
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=self.num_labels)
        self.train(model, output_dir, num_train_epochs)
    
    def train_from_checkpoint(self, checkpoint_dir, completed_steps, total_steps, output_dir):
        model = DistilBertForSequenceClassification.from_pretrained(checkpoint_dir)
        remaining_steps = total_steps - completed_steps
        self.train(model, output_dir, 1, remaining_steps)
    
    def train(self, model, output_dir, num_train_epochs, max_steps=None):
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./zero_logs',
            logging_steps=100,
            save_strategy='steps',
            save_steps=1000,
            evaluation_strategy='epoch',
            gradient_accumulation_steps=2,
            max_steps=max_steps,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.train_dataloader.dataset,
            eval_dataset=self.test_dataloader.dataset,
            data_collator=self.train_dataloader.collate_fn,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()

        model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

# Load and preprocess the data
df = pd.read_pickle('../data/capstone_cleaned_zero.pkl')

# Train Test Split
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Encode the labels
le = LabelEncoder()
train_data.iloc[:, -1] = le.fit_transform(train_data.iloc[:, -1])
test_data.iloc[:, -1] = le.transform(test_data.iloc[:, -1])

# Create a TextClassifier instance
classifier = TextClassifier(train_data, test_data, num_labels=len(le.classes_))

# Train from scratch
output_dir = './zero_results'
classifier.train_from_scratch(output_dir)

# Train from checkpoint
checkpoint_dir = "C:/Users/root/Documents/projects/jupyter/zero_results/checkpoint-84000"
completed_steps = 84000
total_steps = 106152
output_dir = './zero_results_from_checkpoint'
classifier.train_from_checkpoint(checkpoint_dir, completed_steps, total_steps, output_dir)
