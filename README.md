# BERT-based AI Assistant for Text Classification

A versatile deep learning AI assistant that uses PyTorch and BERT to classify text into different intents.

## Features

- 🧠 **BERT-Powered Understanding**: Uses Google's BERT model for state-of-the-art text comprehension
- 🚀 **Performance**: Fine-tuned deep learning models for accurate intent classification
- 📊 **Detailed Analytics**: Comprehensive performance metrics and visualizations
- 🔄 **Easy Training**: Simple API for training on your custom intent data
- 🔮 **Prediction API**: Simple interface for making batch or single predictions
- 📝 **Sample Data Generation**: Built-in functionality to create sample training data

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/bert-assistant.git
   cd bert-assistant
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Download a pre-trained model or train your own as described below.

## Quick Start

```python
from bert_assistant import AIAssistant

# Initialize the assistant
assistant = AIAssistant()

# Create sample data if you don't have your own
sample_data_path = assistant.create_sample_data('sample_intents.json')

# Train the model (only need to do this once)
assistant.train(
    data_path=sample_data_path,
    output_path='bert_intent_classifier.pt',
    batch_size=16,
    num_epochs=3
)

# Make predictions
text = "Remind me to call mom tomorrow"
result = assistant.predict(text)
print(f"Intent: {result['intent']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## Usage Guide

### Training on Custom Data

Prepare your data in CSV or JSON format with 'text' and 'label' columns/fields, then:

```python
assistant = AIAssistant()
assistant.train(
    data_path='your_data.csv',  # or .json
    output_path='your_model.pt',
    batch_size=16,
    num_epochs=5,
    learning_rate=2e-5
)
```

### Loading an Existing Model

```python
assistant = AIAssistant(model_path='your_model.pt')
```

### Making Predictions

For a single prediction:

```python
result = assistant.predict("What's the weather forecast for tomorrow?")
print(f"Intent: {result['intent']}")
print(f"Confidence: {result['confidence']:.2%}")
```

For batch predictions:

```python
texts = [
    "Hello there!",
    "What's the weather like today?",
    "Remind me to buy milk"
]
results = assistant.predict_batch(texts)
for result in results:
    print(f"Text: {result['text']}")
    print(f"Intent: {result['intent']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("---")
```

### Evaluating Model Performance

```python
evaluation = assistant.detailed_evaluation('test_data.csv')
print(f"Accuracy: {evaluation['accuracy']:.2%}")
```

This will also generate a confusion matrix visualization saved as a PNG file.

## Data Format

Your training data should be in one of these formats:

### CSV Format

```csv
text,label
"Hello there!",greeting
"What's the weather like today?",weather
"Remind me to buy milk",reminder
```

### JSON Format

```json
[
  {"text": "Hello there!", "label": "greeting"},
  {"text": "What's the weather like today?", "label": "weather"},
  {"text": "Remind me to buy milk", "label": "reminder"}
]
```

## Model Architecture

The AI assistant uses a fine-tuned BERT model with the following architecture:

1. **BERT Base Layer**: Pre-trained BERT model that understands language context
2. **Dropout Layer**: Prevents overfitting during training
3. **Classification Layer**: Linear layer that maps BERT's output to intent classes

## Performance Optimization

For better performance:

- Use GPU acceleration if available (automatically detected)
- Adjust batch size based on your available memory
- Increase `num_epochs` for more training iterations
- Try different learning rates (2e-5 to 5e-5 usually works well)

## Requirements

See `requirements.txt` for the complete list of dependencies.

## Advanced Usage

### Creating a Custom Dataset Programmatically

```python
custom_data = [
    {"text": "Book a flight to London", "label": "book_flight"},
    {"text": "I want to fly to Paris", "label": "book_flight"},
    {"text": "What's the status of my order?", "label": "order_status"},
    {"text": "I need to cancel my order", "label": "cancel_order"}
]

with open('custom_intents.json', 'w') as f:
    json.dump(custom_data, f)

assistant.train(data_path='custom_intents.json')
```

### Hyperparameter Tuning

The default hyperparameters work well for most cases, but you can adjust them:

```python
assistant.train(
    data_path='your_data.csv',
    batch_size=32,  # Increase for faster training (if memory allows)
    num_epochs=10,  # More epochs for better convergence
    learning_rate=3e-5,  # Adjust learning rate
    max_length=64  # Shorter sequences for faster training
)
```

## License

[MIT License](LICENSE)

## Acknowledgments

- This project uses the Hugging Face Transformers library for BERT implementation
- BERT was originally developed by Google Research