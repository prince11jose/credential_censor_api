import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi import FastAPI, Request
import uvicorn
import re
import logging
import json

# Set up logging
logging.basicConfig(filename='credential_detection.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# # Step 1: Set up the tokenizer and model (If transformers-cli is configured)
# model_name = "Qwen/Qwen2.5-0.5B"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, device_map="auto", low_cpu_mem_usage=True)

# Step 1: Set up the tokenizer and model
model_path = "./local_qwen_2.5-0.5b"  # Path to the downloaded local model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2, device_map="auto", low_cpu_mem_usage=True)


# Step 2: Define the credential detection function with enhanced pattern matching and context awareness
def detect_credentials(text):
    # Define patterns for possible credential indicators with values
    credential_value_patterns = [
        r'(?i)\b(username|usrname|user|login|credential|acct|account)\s*[:=\-\s]+\S+',
        r'(?i)\b(password|pwd|pass|secret|key|token|auth|authentication|pin|pincode)\s*[:=\-\s]+\S+',
        r'(?i)\b(api[_-]?key|access[_-]?key|secret[_-]?key|private[_-]?key)\s*[:=\-\s]+\S+',
        r'(?i)\b(session[_-]?id|session[_-]?token|auth[_-]?token|oauth[_-]?token)\s*[:=\-\s]+\S+'
    ]

    # Define patterns for credential-related keywords without values
    credential_keyword_patterns = [
        r'(?i)\b(username|usrname|user|login|credential|acct|account)\b',
        r'(?i)\b(password|pwd|pass|secret|key|token|auth|authentication|pin|pincode)\b',
        r'(?i)\b(api[_-]?key|access[_-]?key|secret[_-]?key|private[_-]?key)\b',
        r'(?i)\b(session[_-]?id|session[_-]?token|auth[_-]?token|oauth[_-]?token)\b'
    ]

    # Check if there are credential patterns with values in the text
    contains_pattern_with_value = False
    for pattern in credential_value_patterns:
        if re.search(pattern, text):
            contains_pattern_with_value = True
            break

    # Check if there are credential-related keywords without values in the text
    contains_keyword_only = False
    if not contains_pattern_with_value:
        for pattern in credential_keyword_patterns:
            if re.search(pattern, text):
                contains_keyword_only = True
                break

    # Context-aware analysis using the model
    inputs = tokenizer(text, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.softmax(outputs.logits, dim=1).squeeze()
        positive_score = scores[1].item()
        negative_score = scores[0].item()

    # Determine if the text contains credentials with values, considering both heuristic and model results
    if contains_pattern_with_value:
        adjusted_score = min(positive_score + 0.3, 1.0)  # Reduce the score increase if value patterns are present
        return ("Positive", adjusted_score) if adjusted_score > 0.7 else ("Negative", adjusted_score)  # Ensure threshold for Positive
    elif contains_keyword_only:
        adjusted_score = min(positive_score + 0.1, 1.0)  # Smaller increase if only keywords are present without values
        return "Negative", adjusted_score
    elif positive_score > negative_score:
        return "Negative", positive_score  # Lower confidence if no values are detected
    else:
        return "Negative", negative_score

# Step 3: Set up the FastAPI application
app = FastAPI()

@app.post("/detect_credentials")
async def detect_credentials_api(request: Request):
    data = await request.json()
    text = data.get("text")
    if not text:
        return {"error": "Text field is required"}
    result, score = detect_credentials(text)

    # Log the input and output
    logging.info(f"Input: {text}, Result: {result}, Score: {score}")

    # Save the input and output to a file (or a database)
    with open("training_data.json", "a") as f:
        json.dump({"text": text, "result": result, "score": score}, f)
        f.write("\n")
      
    return {"result": result, "score": score}

# Step 4: Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
