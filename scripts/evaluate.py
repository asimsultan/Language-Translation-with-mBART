import torch
from transformers import MBartForConditionalGeneration, MBartTokenizer
from datasets import load_dataset
from utils import get_device, tokenize_data, create_data_loader

# Parameters
model_dir = './models'
max_length = 128
batch_size = 8

# Load Model and Tokenizer
model = MBartForConditionalGeneration.from_pretrained(model_dir)
tokenizer = MBartTokenizer.from_pretrained(model_dir)

# Device
device = get_device()
model.to(device)

# Load Dataset
dataset = load_dataset('wmt16', 'ro-en')
validation_dataset = dataset['validation']

# Tokenize Data
validation_dataset = validation_dataset.map(lambda x: tokenize_data(x, tokenizer, max_length), batched=True)
validation_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# DataLoader
validation_loader = create_data_loader(validation_dataset, batch_size, SequentialSampler)

# Evaluation Function
def evaluate(model, data_loader, device, tokenizer):
    model.eval()
    total_loss = 0
    predictions = []
    references = []

    with torch.no_grad():
        for batch in data_loader:
            b_input_ids = batch['input_ids'].to(device)
            b_attention_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)

            outputs = model(
                input_ids=b_input_ids,
                attention_mask=b_attention_mask,
                labels=b_labels
            )
            loss = outputs.loss
            total_loss += loss.item()

            preds = model.generate(input_ids=b_input_ids, attention_mask=b_attention_mask, max_length=max_length)
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(b_labels, skip_special_tokens=True)

            predictions.extend(decoded_preds)
            references.extend(decoded_labels)

    avg_loss = total_loss / len(data_loader)
    return avg_loss, predictions, references

# Evaluate
validation_loss, predictions, references = evaluate(model, validation_loader, device, tokenizer)
print(f'Validation Loss: {validation_loss}')

# Save the predictions and references
output_dir = './outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(os.path.join(output_dir, 'predictions.txt'), 'w') as f:
    for pred in predictions:
        f.write(pred + '\n')

with open(os.path.join(output_dir, 'references.txt'), 'w') as f:
    for ref in references:
        f.write(ref + '\n')
