from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Trainer, TrainingArguments
from datasets import load_from_disk

model_name = "google/byt5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def tokenize_fn(batch):
    model_inputs = tokenizer(batch["src"], padding="max_length", truncation=True, max_length=256)

    labels = tokenizer(batch["tgt"], padding="max_length", truncation=True, max_length=256)

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs



train_ds = load_from_disk("hf_train_ds")
test_ds  = load_from_disk("hf_test_ds")


split = train_ds.train_test_split(test_size=0.1,   # 10% validation
    seed=42
)

train_ds_new = split["train"]
val_ds   = split["test"]


tokenized_train = train_ds.map(tokenize_fn, batched=True, remove_columns=train_ds.column_names)

tokenized_val = val_ds.map(tokenize_fn,batched=True, remove_columns=val_ds.column_names)


tokenized_train.save_to_disk("hf_tokenized_train")
tokenized_val.save_to_disk("hf_tokenized_val")

print("✅ Tokenization done & saved")


print(tokenized_train[0].keys())

print(tokenizer.decode(tokenized_train[0]["input_ids"][:100]))
print(tokenizer.decode([x for x in tokenized_train[0]["labels"] if x != -100]))