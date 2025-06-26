# Clinical-Topic-Modeling
A comprehensive framework for transforming unstructured clinical data into interpretable, multi-level textual representations for medical prediction tasks using BERT-based topic modeling and hierarchical feature extraction.

## 🎯 Overview

This framework enables:
- **Clinical Topic Modeling**: Transform continuous clinical variables into interpretable categorical tokens
- **Multi-level Feature Extraction**: Extract features at word, sentence, and paragraph levels
- **BERT-based Classification**: Enhanced BERT architecture with hierarchical pooling
- **Flexible Evaluation**: Support for both binary and multi-class classification tasks

## 📋 Features

- ✅ Configurable clinical topic extraction
- ✅ Multi-granularity text processing (word/sentence/paragraph level)
- ✅ BERT-based embedding with hybrid pooling strategy
- ✅ Extensive classifier comparison (10+ models including classical, ensemble, and BERT-based)
- ✅ Cross-validation evaluation framework
- ✅ Interpretable clinical predictions
- ✅ Modular and extensible design

## 🚀 Quick Start

### Basic Usage

```python
from src.data.data_loader import ClinicalDataLoader
from src.data.topic_extractor import ClinicalTopicExtractor
from src.models.bert_classifier import BERTTopicClassifier
from src.evaluation.evaluator import ModelEvaluator

# 1. Load and preprocess data
loader = ClinicalDataLoader('config/config.yaml')
data = loader.load_data('path/to/your/dataset.csv')

# 2. Extract clinical topics
topic_extractor = ClinicalTopicExtractor()
clinical_texts = topic_extractor.create_clinical_text(data)

# 3. Train BERT-based model
model = BERTTopicClassifier()
model.fit(clinical_texts, labels)

# 4. Evaluate model
evaluator = ModelEvaluator()
results = evaluator.evaluate_model(model, test_data, test_labels)
```

### Custom Dataset Example

```python
# Configure for your specific clinical domain
config = {
    'clinical_features': [
        'feature1_max', 'feature2_min', 'feature3_avg'
    ],
    'topic_definitions': {
        'condition1': ('feature1_max', '>', 1.5),
        'condition2': ('feature2_min', '<', 90),
        'condition3': ('feature3_avg', '>', 5.0)
    },
    'extraction_levels': ['word', 'sentence', 'paragraph'],
    'model_type': 'bert_topic_classifier'
}

# Use the framework with your configuration
framework = ClinicalTopicModelingFramework(config)
results = framework.run_complete_evaluation(your_data, your_labels)
```

## 📊 Data Format

Your dataset should be in CSV format with the following structure:

```csv
patient_id,feature1_max,feature2_min,feature3_avg,target_label
1,1.2,95,4.5,0
2,2.1,85,6.2,1
3,0.9,100,3.8,0
...
```

### Required Columns:
- **Patient ID**: Unique identifier for each patient
- **Clinical Features**: Numerical values for clinical measurements
- **Target Label**: Classification target (binary: 0/1 or multi-class: 0/1/2)

### Optional Columns:
- **Temporal Features**: Time-series clinical data
- **Categorical Features**: Pre-existing categorical clinical indicators

## ⚙️ Configuration

### Main Configuration (`config/config.yaml`)

```yaml
# Data Configuration
data:
  input_file: "data/your_dataset.csv"
  target_column: "target_label"
  patient_id_column: "patient_id"
  
# Clinical Topic Configuration
clinical_topics:
  enable: true
  feature_columns:
    - "feature1_max"
    - "feature2_min" 
    - "feature3_avg"
  
  topic_definitions:
    elevated_feature1:
      column: "feature1_max"
      operator: ">"
      threshold: 1.2
    reduced_feature2:
      column: "feature2_min"
      operator: "<"
      threshold: 90
    
# Feature Extraction
feature_extraction:
  levels: ["word", "sentence", "paragraph"]
  word_extraction_sizes: [5, 10, 15, 20]
  sentence_extraction_sizes: [1, 2, 3]
  paragraph_extraction_sizes: [0, 1, 2]

# Model Configuration  
models:
  bert_model_name: "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
  max_length: 128
  batch_size: 16
  learning_rate: 2e-5
  num_epochs: 3

# Evaluation
evaluation:
  cv_folds: 5
  random_state: 42
  metrics: ["accuracy", "precision", "recall", "f1"]
```

## 🤖 Supported Models

### Classical Models
- Logistic Regression
- Support Vector Machines (SVC, LinearSVC)
- Naive Bayes (Multinomial, Bernoulli, Complement)
- Decision Trees
- K-Nearest Neighbors
- Neural Networks (MLP)

### Ensemble Models
- Random Forest
- Gradient Boosting
- Extra Trees
- AdaBoost
- XGBoost (optional)
- LightGBM (optional)
- CatBoost (optional)

### BERT-based Models
- Custom BERT with Hierarchical Pooling

## 📈 Evaluation Metrics

The framework supports comprehensive evaluation:

- **Classification Metrics**: Accuracy, Precision, Recall, F1-score
- **ROC Analysis**: AUC-ROC, AUC-PRC
- **Cross-validation**: Stratified K-fold
- **Statistical Analysis**: Confidence intervals, significance testing

## 🔧 Customization

### Adding Custom Clinical Topics

```python
def create_custom_topics(row_dict):
    """Define your own clinical topic extraction logic"""
    features = []
    
    # Example: Custom condition detection
    if row_dict.get('custom_feature', 0) > threshold:
        features.append('custom_condition')
    
    # Add more conditions as needed
    
    return ' '.join(features) if features else 'normal'

# Register custom topic extractor
topic_extractor = ClinicalTopicExtractor()
topic_extractor.set_custom_extractor(create_custom_topics)
```

### Adding Custom Models

```python
from src.models.base_model import BaseClassifier

class CustomClassifier(BaseClassifier):
    def __init__(self, **kwargs):
        super().__init__()
        # Initialize your custom model
        
    def fit(self, X, y):
        # Implement training logic
        pass
        
    def predict(self, X):
        # Implement prediction logic
        pass

# Register with framework
framework.register_model('custom_model', CustomClassifier)
```

## 📚 Examples

### Example 1: Basic Binary Classification

```python
import pandas as pd
from src import ClinicalTopicModelingFramework

# Load your data
data = pd.read_csv('your_clinical_data.csv')

# Initialize framework
framework = ClinicalTopicModelingFramework('config/config.yaml')

# Run evaluation
results = framework.evaluate_binary_classification(
    data=data,
    target_column='binary_label',
    extraction_methods=['word', 'sentence']
)

# Display results
framework.display_results(results)
```

### Example 2: Multi-class Risk Stratification

```python
# Configure for 3-class problem
config = framework.config.copy()
config['evaluation']['task_type'] = 'multiclass'
config['evaluation']['num_classes'] = 3

# Run evaluation
results = framework.evaluate_multiclass_classification(
    data=data,
    target_column='risk_level',  # 0: Low, 1: Medium, 2: High
    class_names=['Low Risk', 'Medium Risk', 'High Risk']
)
```

### Example 3: Custom Feature Engineering

```python
# Define custom clinical features
def extract_custom_features(df):
    """Extract domain-specific clinical features"""
    df['risk_score'] = (df['feature1'] * 0.3 + 
                       df['feature2'] * 0.5 + 
                       df['feature3'] * 0.2)
    
    df['severity_category'] = pd.cut(df['risk_score'], 
                                   bins=[0, 0.3, 0.7, 1.0], 
                                   labels=['mild', 'moderate', 'severe'])
    return df

# Use with framework
framework.add_feature_engineer(extract_custom_features)
results = framework.run_evaluation(data)
```

## 🧪 Testing

Run tests to ensure everything works correctly:

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_data_processing.py
python -m pytest tests/test_models.py
python -m pytest tests/test_evaluation.py

# Run with coverage
python -m pytest tests/ --cov=src/
```

## 📊 Benchmarking

Compare different approaches:

```python
from src.evaluation.benchmark import ModelBenchmark

benchmark = ModelBenchmark()

# Add models to compare
benchmark.add_model('bert_topic', BERTTopicClassifier())
benchmark.add_model('random_forest', RandomForestClassifier())
benchmark.add_model('svm', SVC())

# Run benchmark
results = benchmark.run_benchmark(
    data=your_data,
    labels=your_labels,
    cv_folds=5
)

# Generate comparison report
benchmark.generate_report(results, output_path='benchmark_results.html')
```

## 🔍 Interpretability

Generate interpretable insights:

```python
from src.evaluation.interpretability import ModelInterpreter

interpreter = ModelInterpreter()

# Get feature importance
importance = interpreter.get_feature_importance(model, feature_names)

# Generate clinical insights
insights = interpreter.generate_clinical_insights(
    model=model,
    data=test_data,
    feature_names=clinical_features
)

# Create visualization
interpreter.plot_feature_importance(importance, save_path='feature_importance.png')
```

## 📝 Citation

If you use this framework in your research, please cite:

```bibtex
@article{clinical_topic_modeling_2025,
  title={Integrating Clinical Topic Modeling: A BERT-Based Framework for Medical Risk Prediction},
  author={Your Name},
  journal={Conference},
  year={2025},
  doi={your-doi}
}
```

## 🤝 Contributing

We welcome contributions! Thank You.

## 🙏 Acknowledgments

- Inspired by clinical NLP research and topic modeling techniques
- Thanks to the open-source community for the foundational libraries

---

**Happy modeling! 🎉**
