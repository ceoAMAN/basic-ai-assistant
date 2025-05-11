import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntentClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes, dropout_rate=0.1):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits

class AIAssistant:
    def __init__(self, model_path=None, intent_labels_path=None, bert_model_name='bert-base-uncased'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        
       
        if intent_labels_path and os.path.exists(intent_labels_path):
            with open(intent_labels_path, 'r') as f:
                self.intent_labels = json.load(f)
            self.label2id = {label: i for i, label in enumerate(self.intent_labels)}
            self.id2label = {i: label for i, label in enumerate(self.intent_labels)}
            self.num_labels = len(self.intent_labels)
            logger.info(f"Loaded {self.num_labels} intent labels")
        else:
            self.intent_labels = None
            self.label2id = None
            self.id2label = None
            self.num_labels = None
        
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        elif self.num_labels:
            self.model = BERTClassifier(bert_model_name, self.num_labels)
            self.model.to(self.device)
        else:
            self.model = None
            logger.warning("No model loaded and no intent labels provided")
    
    def load_model(self, model_path):
        """Load a trained model from a file."""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.num_labels = checkpoint.get('num_labels', self.num_labels)
        self.intent_labels = checkpoint.get('intent_labels', self.intent_labels)
        self.label2id = checkpoint.get('label2id', self.label2id)
        self.id2label = checkpoint.get('id2label', self.id2label)
        
       
        bert_model_name = checkpoint.get('bert_model_name', 'bert-base-uncased')
        self.model = BERTClassifier(bert_model_name, self.num_labels)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded from {model_path}")
    
    def save_model(self, model_path):
        """Save the trained model to a file."""
        if not self.model:
            logger.error("No model to save")
            return
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'num_labels': self.num_labels,
            'intent_labels': self.intent_labels,
            'label2id': self.label2id,
            'id2label': self.id2label,
            'bert_model_name': 'bert-base-uncased'  
        }
        
        torch.save(checkpoint, model_path)
        logger.info(f"Model saved to {model_path}")
    
    def prepare_data(self, data_path, test_size=0.2, random_state=42):
        """
        Prepare data for training from a CSV or JSON file.
        
        Expected format:
        - CSV: columns 'text' and 'label'
        - JSON: list of objects with 'text' and 'label' keys
        """
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        else:
            raise ValueError("Data file must be CSV or JSON")
        
        texts = df['text'].values
        labels_text = df['label'].values
        
        
        if not self.intent_labels:
            self.intent_labels = sorted(list(set(labels_text)))
            self.label2id = {label: i for i, label in enumerate(self.intent_labels)}
            self.id2label = {i: label for i, label in enumerate(self.intent_labels)}
            self.num_labels = len(self.intent_labels)
            logger.info(f"Created {self.num_labels} intent labels")
        
       
        labels = np.array([self.label2id[label] for label in labels_text])
        
       
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        return X_train, X_test, y_train, y_test
    
    def train(self, data_path, output_path='bert_classifier.pt', batch_size=16, num_epochs=5, learning_rate=2e-5, max_length=128, test_size=0.2, random_state=42):
        """Train the BERT classifier on the provided data."""
       
        X_train, X_test, y_train, y_test = self.prepare_data(
            data_path, test_size=test_size, random_state=random_state
        )
        
        
        train_dataset = IntentClassificationDataset(
            X_train, y_train, self.tokenizer, max_length=max_length
        )
        test_dataset = IntentClassificationDataset(
            X_test, y_test, self.tokenizer, max_length=max_length
        )
        
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        
        if not self.model:
            self.model = BERTClassifier('bert-base-uncased', self.num_labels)
            self.model.to(self.device)
        
        
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
       
        best_accuracy = 0.0
        
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            
           
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in progress_bar:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask, token_type_ids)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': total_loss/(progress_bar.n+1)})
            
          
            accuracy = self.evaluate(test_loader)
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}, Test Accuracy: {accuracy:.4f}")
            
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.save_model(output_path)
                logger.info(f"New best model saved with accuracy: {best_accuracy:.4f}")
        
       
        self.load_model(output_path)
        return best_accuracy
    
    def evaluate(self, test_loader=None, test_data=None):
        """Evaluate the model on test data."""
        if not self.model:
            logger.error("No model available for evaluation")
            return 0.0
        
        self.model.eval()
        
        
        if test_loader is None and test_data is not None:
            if isinstance(test_data, tuple) and len(test_data) == 2:
                texts, labels = test_data
                test_dataset = IntentClassificationDataset(
                    texts, labels, self.tokenizer, max_length=128
                )
                test_loader = DataLoader(test_dataset, batch_size=16)
            else:
                logger.error("Invalid test data format")
                return 0.0
        
        if test_loader is None:
            logger.error("No test data provided")
            return 0.0
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask, token_type_ids)
                _, predictions = torch.max(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_predictions)
        return accuracy
    
    def detailed_evaluation(self, test_data_path):
        """Provide a detailed evaluation report with classification metrics."""
        if not self.model:
            logger.error("No model available for evaluation")
            return None
        
       
        if test_data_path.endswith('.csv'):
            df = pd.read_csv(test_data_path)
        elif test_data_path.endswith('.json'):
            with open(test_data_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        else:
            logger.error("Test data file must be CSV or JSON")
            return None
        
        texts = df['text'].values
        labels_text = df['label'].values
        
        
        labels = np.array([self.label2id.get(label, 0) for label in labels_text])
        
        
        test_dataset = IntentClassificationDataset(
            texts, labels, self.tokenizer, max_length=128
        )
        test_loader = DataLoader(test_dataset, batch_size=16)
        
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask, token_type_ids)
                _, predictions = torch.max(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        
        pred_labels = [self.id2label[pred] for pred in all_predictions]
        true_labels = [self.id2label[label] for label in all_labels]
        
       
        accuracy = accuracy_score(true_labels, pred_labels)
        report = classification_report(true_labels, pred_labels, output_dict=True)
        conf_matrix = confusion_matrix(true_labels, pred_labels, labels=self.intent_labels)
        
       
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            conf_matrix, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.intent_labels,
            yticklabels=self.intent_labels
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
       
        plot_path = os.path.splitext(test_data_path)[0] + '_confusion_matrix.png'
        plt.savefig(plot_path)
        
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Classification Report:\n{classification_report(true_labels, pred_labels)}")
        logger.info(f"Confusion matrix saved to {plot_path}")
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist(),
            'plot_path': plot_path
        }
    
    def predict(self, text):
        """Predict the intent of a single text input."""
        if not self.model:
            logger.error("No model available for prediction")
            return None
        
        self.model.eval()
        
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
       
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        token_type_ids = encoding['token_type_ids'].to(self.device)
        
       
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask, token_type_ids)
            _, prediction = torch.max(outputs, dim=1)
        
        predicted_label_id = prediction.item()
        predicted_label = self.id2label[predicted_label_id]
        
        
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence = probs[0][predicted_label_id].item()
        
        return {
            'intent': predicted_label,
            'confidence': confidence,
            'all_probs': dict(zip(self.intent_labels, probs[0].cpu().numpy().tolist()))
        }
    
    def predict_batch(self, texts):
        """Predict intents for a batch of texts."""
        if not self.model:
            logger.error("No model available for prediction")
            return None
        
        self.model.eval()
        
        
        encodings = self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
       
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        token_type_ids = encodings['token_type_ids'].to(self.device)
        
       
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask, token_type_ids)
            _, predictions = torch.max(outputs, dim=1)
        
       
        probs = torch.nn.functional.softmax(outputs, dim=1)
        
        results = []
        for i, pred in enumerate(predictions):
            predicted_label_id = pred.item()
            predicted_label = self.id2label[predicted_label_id]
            confidence = probs[i][predicted_label_id].item()
            
            results.append({
                'text': texts[i],
                'intent': predicted_label,
                'confidence': confidence,
                'all_probs': dict(zip(self.intent_labels, probs[i].cpu().numpy().tolist()))
            })
        
        return results
    
    def create_sample_data(self, output_path='sample_intents.json', num_samples=100):
        """Create sample data for demonstration purposes."""
        sample_intents = {
            'greeting': [
                'Hello there!',
                'Hi, how are you?',
                'Good morning!',
                'Hey, nice to meet you',
                'Greetings, how can you help me?',
                'Hello, how are you doing today?',
                'Hi there, I need some assistance',
                'Hey, what\'s up?',
                'Good afternoon, I have a question',
                'Hello AI assistant'
            ],
            'weather': [
                'What\'s the weather like today?',
                'Is it going to rain tomorrow?',
                'Tell me the forecast for this weekend',
                'How hot will it be today?',
                'What\'s the temperature outside?',
                'Will I need an umbrella today?',
                'Is it sunny outside?',
                'What\'s the weather forecast?',
                'How cold is it going to be tomorrow?',
                'Will there be snow this week?'
            ],
            'reminder': [
                'Set a reminder for my meeting at 3pm',
                'Remind me to call mom tomorrow',
                'Can you remind me to take my medication?',
                'I need a reminder for my dentist appointment',
                'Set an alarm for 7am',
                'Don\'t let me forget to submit my report',
                'Remind me to pick up groceries after work',
                'Set a reminder for my anniversary next week',
                'Can you remind me to water the plants?',
                'I need to remember to pay my bills on Friday'
            ],
            'search': [
                'Search for Italian restaurants nearby',
                'Find information about quantum computing',
                'Look up the definition of photosynthesis',
                'Search for flights to New York',
                'Find a recipe for chocolate cake',
                'Search for news about climate change',
                'Find the nearest gas station',
                'Look up the capital of Australia',
                'Search for reviews of the new iPhone',
                'Find information about the French Revolution'
            ]
        }
        
        data = []
        for intent, examples in sample_intents.items():
            for _ in range(num_samples // len(sample_intents)):
                for example in examples:
                    data.append({
                        'text': example,
                        'label': intent
                    })
        
       
        np.random.shuffle(data)
        
       
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Created sample data with {len(data)} examples at {output_path}")
        return output_path


def main():
   
    assistant = AIAssistant()
    
    
    sample_data_path = 'sample_intents.json'
    if not os.path.exists(sample_data_path):
        assistant.create_sample_data(sample_data_path)
    
   
    assistant.train(
        data_path=sample_data_path,
        output_path='bert_intent_classifier.pt',
        batch_size=16,
        num_epochs=3
    )
    
   
    evaluation_results = assistant.detailed_evaluation(sample_data_path)
    
   
    test_texts = [
        "Hello, how are you doing?",
        "What's the weather today?",
        "Remind me to buy milk tomorrow",
        "Can you find information about climate change?"
    ]
    
    print("\nPredictions:")
    for text in test_texts:
        result = assistant.predict(text)
        print(f"Text: {text}")
        print(f"Predicted Intent: {result['intent']} (Confidence: {result['confidence']:.4f})")
        print()

if __name__ == "__main__":
    main()