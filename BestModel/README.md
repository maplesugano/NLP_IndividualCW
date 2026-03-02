# DeBERTa-v3-base Model for PCL Classification

This folder contains the implementation of the proposed approach for detecting Patronizing and Condescending Language (PCL) in text.

## Proposed Approach

The model implements a DeBERTa-v3-base architecture with enhanced training strategies:

- **Model**: microsoft/deberta-v3-base (184M parameters)
- **Max Sequence Length**: 192 tokens (optimized based on EDA)
- **Training Epochs**: 4-5 with early stopping on F1 score
- **Class Weighting**: Inverse frequency weighting (1.53:1 for Non-PCL:PCL)
- **Learning Rate**: 2e-5 with warmup (10%) and linear decay
- **Gradient Accumulation**: 2 steps
- **Batch Size**: 16

## Key Improvements Over Baseline

1. **Superior Architecture**: DeBERTa-v3's disentangled attention mechanism better captures subtle linguistic patterns
2. **Full Dataset Training**: Uses all 8,375 training samples with class weighting instead of aggressive downsampling
3. **Extended Training**: 4-5 epochs with early stopping vs. baseline's 1 epoch
4. **Optimized Sequence Length**: 192 tokens covers 95%+ of samples while reducing computational overhead
5. **Advanced Optimization**: Learning rate warmup and linear decay for stable training

## Files

- `train_deberta_model.ipynb`: Complete training notebook with all implementation details
- `requirements.txt`: Python dependencies required to run the code
- `README.md`: This file

## Installation

1. Clone or download this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Training the Model

### Option 1: Using Jupyter Notebook

1. Open `train_deberta_model.ipynb` in Jupyter Lab/Notebook or VS Code
2. Run all cells sequentially
3. The notebook will:
   - Load and preprocess the PCL dataset
   - Train the DeBERTa-v3-base model
   - Evaluate on the dev set
   - Save the best model

### Option 2: Using Google Colab

1. Upload `train_deberta_model.ipynb` to Google Colab
2. Run all cells
3. For faster training, enable GPU: Runtime → Change runtime type → GPU

## Expected Outputs

After training completes, you will have:

1. **Best Model**: `best_deberta_model.pt` - PyTorch checkpoint of the best model
2. **Full Model**: `./deberta_pcl_model/` directory containing:
   - Model weights
   - Tokenizer configuration
   - Training configuration (JSON)
3. **Training History Plot**: `training_history.png` - Visualization of training metrics
4. **Console Output**: Detailed training logs including:
   - Loss, accuracy, F1 score per epoch
   - Final evaluation metrics on dev set
   - Classification report

## Expected Performance

Based on the proposed approach, we expect:

- **Target Dev F1**: 0.58-0.62 (compared to baseline 0.48)
- **Improvement**: +0.10-0.14 F1 points (~20-29% relative improvement)

## Model Usage

After training, to load and use the model:

```python
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
import torch

# Load model and tokenizer
model = DebertaV2ForSequenceClassification.from_pretrained('./deberta_pcl_model')
tokenizer = DebertaV2Tokenizer.from_pretrained('./deberta_pcl_model')

# Prepare input text
text = "Your text here"
inputs = tokenizer(text, return_tensors='pt', max_length=192, 
                   padding='max_length', truncation=True)

# Get prediction
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()

# 0 = Non-PCL, 1 = PCL
print(f"Prediction: {'PCL' if prediction == 1 else 'Non-PCL'}")
```

## Training Time Estimates

- **With GPU (NVIDIA T4 or better)**: ~2-3 hours
- **With CPU**: ~8-12 hours (not recommended)

## Hardware Requirements

- **Minimum**: 8GB RAM, 2GB GPU memory
- **Recommended**: 16GB RAM, 8GB+ GPU memory (e.g., NVIDIA T4, V100, or A100)

## Troubleshooting

### Out of Memory Errors

If you encounter OOM errors:
1. Reduce `BATCH_SIZE` from 16 to 8 or 4
2. Reduce `MAX_LENGTH` from 192 to 128
3. Increase `GRADIENT_ACCUMULATION_STEPS` to 4

### Slow Training

If training is too slow:
1. Use Google Colab with GPU enabled
2. Reduce `NUM_EPOCHS` from 5 to 3
3. Use a smaller model like `microsoft/deberta-v3-small` (but expect lower performance)

## Implementation Details

### Data Processing
- Loads data from official GitHub repositories
- Merges train/dev splits with main dataset
- Applies class-weighted loss function

### Model Architecture
- DeBERTa-v3-base encoder (12 layers, 768 hidden dimensions)
- Disentangled attention mechanism
- Dropout layer (p=0.1)
- Binary classification head

### Training Strategy
- AdamW optimizer with weight decay (0.01)
- Linear learning rate schedule with warmup
- Gradient clipping (max_norm=1.0)
- Early stopping with patience=2
- Saves best model based on dev F1 score

## References

- DeBERTa-v3 Paper: [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2111.09543)
- Original PCL Dataset: [Don't Patronize Me! SemEval-2022 Task 4](https://github.com/Perez-AlmendrosC/dontpatronizeme)

## License

This implementation is for educational purposes as part of NLP coursework.

## Author

Implementation based on the proposed approach in Exercise 3 for detecting Patronizing and Condescending Language.
