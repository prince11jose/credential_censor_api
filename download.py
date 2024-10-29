from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained("./local_qwen_2.5-0.5b")
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.save_pretrained("./local_qwen_2.5-0.5b")
