## LLM-Pipeline-TOD

A Large Language Model-enhanced pipeline framework for task-oriented dialogue systems, designed to reduce error propagation and improve domain adaptation.

## 📋 Project Overview

This project implements a complete Task-Oriented Dialogue (TOD) system pipeline with three core modules:

1.  **Domain Recognition** - Identifies the domains involved in user intent
2.  **State Extraction** - Extracts key information slots from conversations
3.  **Response Generation** - Generates appropriate responses based on extracted states

## ✨ Key Features

*   🔍 **Multi-domain Support**: Supports multiple domains including restaurant, hotel, train, taxi, attraction, hospital, police
*   🤖 **LLM Enhancement**: Fine-tuned on Qwen3-4B model to improve module performance
*   🛠️ **Tool Integration**: Integrated database query and ontology lookup tools
*   📊 **Visualization Analysis**: Provides training process visualization and performance comparison analysis
*   🎯 **Error Propagation Control**: Reduces error accumulation through pipeline design

## 🏗️ System Architecture

```plaintext
User Input → Domain Recognition → State Extraction → Response Generation → System Response
    ↓              ↓                    ↓                    ↓
Vector Search   LLM Classification   LLM Extraction     LLM Generation
    ↓              ↓                    ↓                    ↓
Similar Cases   Domain Labels       Slot Information    Natural Response
```

## 📦 Installation Guide

### Requirements

*   Python 3.8+
*   CUDA 11.8+ (for GPU training)
*   At least 16GB VRAM (recommended A10G or higher)

### Installation Steps

**Clone the repository**

**Install dependencies**

**Download data**

## 🚀 Quick Start

### 1\. Data Preparation

```plaintext
# Generate training data pairs
python generate_train_pairs.py

# Generate validation prompts
python generate_validation_prompts.py
```

### 2\. Model Training

```plaintext
# Train domain recognition model
python finetune_llm.py --train_file domain_recognition_train.jsonl

# Train state extraction model
python finetune_llm.py --train_file state_extraction_train.jsonl

# Train response generation model
python finetune_llm.py --train_file response_generation_train.jsonl
```

### 3\. Model Inference

```plaintext
# Use trained model for inference
python infer_llm.py --model_path qwen3_lora_output/checkpoint-720
```

### 4\. Dialogue Agent

```plaintext
# Start dialogue agent
python agent.py
```

## 📁 Project Structure

```plaintext
llm-pipeline-tod/
├── agent.py                          # Main dialogue agent program
├── finetune_llm.py                   # LLM fine-tuning script
├── generate_train_pairs.py           # Training data generation
├── compare_training_results.py       # Training results comparison analysis
├── multiwoz_utils/                   # MultiWOZ data processing utilities
│   ├── data_loader.py               # Data loader
│   ├── dialog_iterator.py           # Dialogue iterator
│   └── database.py                  # Database query
├── multiwoz_database/               # MultiWOZ database files
├── qwen3_lora_output/               # Trained model outputs
├── plots/                           # Training process visualization charts
└── case_analysis/                   # Case analysis results
```

## 🔧 Core Modules

### Domain Recognition

*   **Function**: Identifies the domains involved in current user dialogue
*   **Input**: Dialogue history and current user input
*   **Output**: Domain labels (restaurant, hotel, train, etc.)
*   **Training Data**: `domain_recognition_train.jsonl`

### State Extraction

*   **Function**: Extracts key information slots from dialogue
*   **Input**: Dialogue history, user input, domain information
*   **Output**: Structured state information (slot-value pairs)
*   **Training Data**: `state_extraction_train.jsonl`

### Response Generation

*   **Function**: Generates natural language responses based on extracted states
*   **Input**: Dialogue history, extracted states, database query results
*   **Output**: System response text
*   **Training Data**: `response_generation_train.jsonl`

## 📊 Performance Analysis

### Training Results Comparison

Run the following command to generate training results comparison charts:

```plaintext
python compare_training_results.py
```

This will generate comprehensive comparison charts including:

*   Training loss curve comparison
*   Learning rate schedule comparison
*   Gradient norm comparison
*   Performance metrics comparison

### Case Analysis

The project includes detailed case analysis located in the `case_analysis/` directory:

*   `perfect_log.txt` - Perfect cases
*   `acceptable_extra_log.txt` - Acceptable extra information cases
*   `missing_log.txt` - Missing information cases
*   `badcase_log.txt` - Error cases

## 🛠️ Tool Integration

### Database Query Tool

```python
# Query hotel information
result = default_database.query("hotel", {"area": "centre", "pricerange": "cheap"})
```

### Ontology Lookup Tool

```python
# Look up possible values for domain slots
values = ontology_lookup("restaurant", "food")
```

## 📈 Training Configuration

### Model Configuration

*   **Base Model**: Qwen/Qwen3-4B
*   **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
*   **Quantization**: 4-bit quantization to save VRAM
*   **Target Modules**: q\_proj, v\_proj

### Training Parameters

*   **Batch Size**: 4 (device batch) × 2 (gradient accumulation) = 8 (effective batch)
*   **Learning Rate**: 2e-4
*   **Training Epochs**: 3
*   **Max Sequence Length**: 512

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

*   [MultiWOZ](https://github.com/budzianowski/multiwoz) - Multi-domain task-oriented dialogue dataset
*   [Qwen](https://github.com/QwenLM/Qwen) - Base language model
*   [Transformers](https://github.com/huggingface/transformers) - Deep learning framework

**Note**: This project is for research purposes only. Please ensure compliance with relevant data usage agreements and model usage terms.  
 

```plaintext
# Extract MultiWOZ database
unzip multiwoz_database.zip
```

```plaintext
pip install -r requirements.txt
```

```plaintext
git clone <repository-url>
cd llm-pipeline-tod
```