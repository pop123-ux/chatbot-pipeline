# Intent-Based Chatbot Pipeline

A complete chatbot system built with TensorFlow/Keras and Flask. The project demonstrates a classic NLP pipeline: training a neural network on intent-pattern data, then serving it through a web interface for real-time conversation.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-000000?logo=flask&logoColor=white)

---

## How It Works

```
┌────────────────────────────────────────────────────────────┐
│                    TRAINING PHASE                          │
│                                                            │
│  intents.json ──→ Tokenize + Lemmatize ──→ Bag-of-Words   │
│                                                 │          │
│                                                 ▼          │
│                                         Dense NN (128→64)  │
│                                                 │          │
│                                                 ▼          │
│                                  chatbot_model.keras       │
│                                  words.pkl / classes.pkl   │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│                   INFERENCE PHASE                          │
│                                                            │
│  User message ──→ Tokenize + Lemmatize ──→ Bag-of-Words   │
│                                                 │          │
│                                                 ▼          │
│                                    Model Prediction        │
│                                                 │          │
│                                                 ▼          │
│                              Intent (confidence > 0.25)    │
│                                                 │          │
│                                                 ▼          │
│                              Random response from intent   │
└────────────────────────────────────────────────────────────┘
```

## Architecture

### Neural Network

A simple fully connected network that classifies user messages into intent categories:

| Layer | Type | Details |
|-------|------|---------|
| Input | Dense | 128 neurons, ReLU activation |
| Dropout | Regularization | 20% dropout |
| Hidden | Dense | 64 neurons, ReLU activation |
| Dropout | Regularization | 20% dropout |
| Output | Dense | N neurons (one per intent), Softmax |

**Training:** 1000 epochs, batch size 5, Adam optimizer (lr=0.001), categorical cross-entropy loss.

### NLP Pipeline

1. **Tokenization** — split input into words using NLTK
2. **Lemmatization** — reduce words to base form (`"running"` → `"run"`)
3. **Bag-of-Words** — binary vector encoding against the vocabulary
4. **Classification** — neural network predicts the most likely intent
5. **Response selection** — random response sampled from the matched intent

## Project Structure

```
chatbot-pipeline/
├── Model Training/
│   ├── model_training.py         # Train the intent classifier
│   └── intents.json              # Intent definitions (patterns + responses)
│
├── Flask Application/
│   ├── app.py                    # Flask web server
│   ├── utils.py                  # Inference utilities (predict + respond)
│   ├── intents.json              # Intent definitions (used at runtime)
│   ├── templates/
│   │   └── index.html            # Chat widget UI
│   └── model/
│       ├── chatbot_model.keras   # Trained model weights
│       ├── words.pkl             # Vocabulary mapping
│       └── classes.pkl           # Intent label mapping
│
└── requirements.txt
```

## Getting Started

### Installation

```bash
git clone https://github.com/pop123-ux/chatbot-pipeline.git
cd chatbot-pipeline
pip install -r requirements.txt
```

### Usage

**Option A — Use the pre-trained model** (fastest):

```bash
cd "Flask Application"
python app.py
```

Open `http://localhost:5000` in your browser and click the chat bubble in the bottom-right corner.

**Option B — Train from scratch:**

```bash
# Step 1: Train the model
cd "Model Training"
python model_training.py

# Step 2: Copy the trained model to the Flask app
cp model/* "../Flask Application/model/"

# Step 3: Start the server
cd "../Flask Application"
python app.py
```

## Intent Categories

The chatbot recognizes **20 intent categories** including:

| Intent | Example Patterns |
|--------|-----------------|
| `greeting` | "Hi", "Hello", "Hey" |
| `ai` | "What is AI?", "Explain artificial intelligence" |
| `programming` | "What is programming?", "Tell me about coding" |
| `cybersecurity` | "What is cybersecurity?", "Why is it important?" |
| `joke` | "Tell me a joke", "Make me laugh" |
| `motivation` | "Motivate me", "I feel lazy" |
| `help` | "Can you help me?", "I have a problem" |
| ... | See `intents.json` for the full list |

To add new intents, edit `intents.json` with new patterns and responses, then retrain the model.

## Key Concepts Demonstrated

- **NLP text preprocessing** — tokenization, lemmatization, and bag-of-words encoding with NLTK
- **Neural network classification** — multi-class intent classification with TensorFlow/Keras
- **Model serialization** — saving/loading models and label encoders with Keras and pickle
- **Web deployment** — serving ML predictions through a Flask REST API
- **Frontend integration** — interactive chat widget with jQuery and AJAX

## License

MIT

## 🔗 More

- Author: [@pop123-ux](https://github.com/pop123-ux)
- Medium write-ups: [medium.com/@Pop123](https://medium.com/@Pop123)

P.S: I usually prefer the PyTorch framework over Tensorflow, this was really a bit of a struggle :)
