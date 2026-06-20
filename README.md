# Qwen 3.5 9B Fine-tuned Model - Replicate Deployment

This repository contains the configuration files to deploy your fine-tuned Qwen 3.5 9B model to Replicate.

## Model Details

- **Base Model**: Qwen 3.5 9B
- **Training Type**: LoRA Fine-tuning
- **Hugging Face Model**: `masidyai/qwen3.5-9b-masidy`
- **Framework**: Transformers

## Files

- `cog.yaml` - Replicate model configuration
- `predict.py` - Inference implementation
- `.gitignore` - Git ignore rules

## Deployment Instructions

### Prerequisites

1. **Replicate Account**: Create one at https://replicate.com
2. **Hugging Face Token**: Get it from https://huggingface.co/settings/tokens
3. **Git LFS**: Install from https://git-lfs.com
4. **Cog CLI**: Install with `pip install cog`

### Step 1: Clone this repository to your local machine

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### Step 2: Test locally (optional but recommended)

```bash
cog predict -i prompt="Hello, how are you?"
```

### Step 3: Push to Replicate

1. Create a model on Replicate dashboard: https://replicate.com/create
2. Follow the push instructions provided by Replicate:

```bash
cog login  # Log in with your Replicate credentials
cog push r8.im/<your-username>/<model-name>
```

### Step 4: Set Hugging Face token as secret (Important!)

1. Go to your model's settings on Replicate
2. Add a secret variable:
   - **Key**: `HF_TOKEN`
   - **Value**: Your Hugging Face API token

This allows the model to download your private Hugging Face model.

## API Usage

Once deployed, use the Replicate API:

```python
import replicate

output = replicate.run(
    "<username>/<model-name>",
    input={
        "prompt": "Your input text here",
        "max_length": 512,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 50,
    }
)
print(output)
```

## Parameters

- **prompt** (string): Input text prompt
- **max_length** (integer): Maximum length of generated text (default: 512)
- **temperature** (float): Sampling temperature 0.0-2.0 (default: 0.7)
- **top_p** (float): Nucleus sampling parameter 0.0-1.0 (default: 0.95)
- **top_k** (integer): Top-k sampling (default: 50)

## Troubleshooting

### Model not loading
- Ensure `HF_TOKEN` is set in Replicate secrets
- Verify model is accessible: https://huggingface.co/masidyai/qwen3.5-9b-masidy

### Out of Memory (OOM)
- Use `load_in_8bit=True` in `predict.py` for 8-bit quantization
- Reduce `max_length` parameter default

### Slow inference
- Model quantization may help (8-bit or 4-bit)
- Consider using smaller max_length values

## Support

For Replicate deployment help: https://replicate.com/docs
For Hugging Face issues: https://huggingface.co/docs
For Qwen model docs: https://huggingface.co/Qwen
