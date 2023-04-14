import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, pipeline
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_metric
import numpy as np

def create_data_loader(data, tokenizer, max_length=512, num_workers=8, pin_memory=True):
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

# Load and preprocess the data
df = pd.read_pickle('../data/capstone_cleaned_zero.pkl')

# Train Test Split
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Encode the labels
le = LabelEncoder()
train_data.iloc[:, -1] = le.fit_transform(train_data.iloc[:, -1])
test_data.iloc[:, -1] = le.transform(test_data.iloc[:, -1])

# Load the model and tokenizer from the checkpoint

checkpoint_dir = "C:/Users/root/Documents/projects/jupyter/zero_results/checkpoint-84000"
model = DistilBertForSequenceClassification.from_pretrained(checkpoint_dir)

# Create the tokenizer and the dataloaders
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased') 

train_dataloader = create_data_loader(train_data.values, tokenizer, num_workers=8, pin_memory=True)
test_dataloader = create_data_loader(test_data.values, tokenizer, num_workers=8, pin_memory=True)


# # Prepare the model
# model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(le.classes_))

# # Set up training arguments
# training_args = TrainingArguments(
#     output_dir='./zero_results',
#     num_train_epochs=3,
#     per_device_train_batch_size=32,
#     per_device_eval_batch_size=32,
#     warmup_steps=500,
#     weight_decay=0.01,
#     logging_dir='./zero_logs',
#     logging_steps=100,
#     save_strategy='steps',
#     save_steps=1000,
#     evaluation_strategy='epoch',
#     gradient_accumulation_steps=2,
# )

####################################################################################################



# Calculate the remaining steps and epochs
completed_steps = 84000
total_steps = 106152
remaining_steps = total_steps - completed_steps

# Update the training arguments for start at 84000
training_args = TrainingArguments(
    output_dir='./zero_results',
    num_train_epochs=1, # Set to 1 epoch since we're resuming with remaining steps
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
    max_steps=remaining_steps, # Add the remaining steps
)
######################################################################################################
# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataloader.dataset,
    eval_dataset=test_dataloader.dataset,
    data_collator=train_dataloader.collate_fn,
    compute_metrics=compute_metrics, 
)

# Train the model
trainer.train()

# Save the model
output_dir = "./zero_fine_tuned"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

