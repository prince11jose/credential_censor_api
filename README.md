# Credential Censor API

## Overview

This project provides a FastAPI-based REST API for detecting potential credentials, such as usernames, passwords, API keys, and authentication tokens in text inputs. It uses a combination of heuristic-based regex patterns and a transformer-based language model for robust detection of sensitive information.

The primary use case is to identify whether a given text contains potential credentials, and return either "Positive" or "Negative" along with a confidence score. This can help in identifying and preventing the exposure of sensitive information.

## Features

Hybrid Approach: Uses regex-based heuristics along with a transformer model (Qwen2.5-0.5B) for detecting credential information.

REST API: Built with FastAPI, the service is easy to interact with and can be integrated with other applications.

Logging and Training Data: Logs every request and response to credential_detection.log, and stores input/output pairs for potential use in model fine-tuning (training_data.json).

## Requirements

Python 3.8+
transformer-cli for downloading model from huggingface

## Libraries:

torch

transformers

fastapi

uvicorn

To install the required dependencies, run:
```
pip install torch transformers fastapi uvicorn
```
## Setup Instructions

1. Download the Model and Tokenizer Locally (Optional)

First, download the model and tokenizer from Hugging Face:
```
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained("./local_qwen_2.5-0.5b")
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.save_pretrained("./local_qwen_2.5-0.5b")
```

2. Run the Application

Save the script as credential_detection.py and run the FastAPI application:
```
python api.py
```
The API will be available at http://localhost:8000.

API Endpoint

POST /detect_credentials

Description: Detects whether a given text contains sensitive information, such as credentials.

Request Body:

text (string): The input text to analyze.

### Response:

result (string): Either "Positive" or "Negative".

score (float): Confidence score for the result.

### Example:
```
curl -X POST "http://localhost:8000/detect_credentials" -H "Content-Type: application/json" -d '{"text": "My username is user123 and password is pass123"}'
```
### Response:
```
{
  "result": "Positive",
  "score": 1.0
}
```
## Author:

Name: Prince

Role: DevOps Engineer with expertise in cloud resource deployment, management, observability, monitoring, and WEB/REST API development.