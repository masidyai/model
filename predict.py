import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM
from cog import BasePredictor, Input, Path
import os

class Predictor(BasePredictor):
    def setup(self):
        """Load model and tokenizer on startup"""
        # Model identifier from Hugging Face
        model_name = "masidyai/qwen3.5-9b-masidy"
        
        # Download and load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=os.getenv("HF_TOKEN")  # Use Hugging Face token if provided
        )
        
        # Load the fine-tuned model with LoRA weights merged
        # Check if model has LoRA adapters
        try:
            self.model = AutoPeftModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                load_in_8bit=False,
                device_map="auto",
                trust_remote_code=True,
                token=os.getenv("HF_TOKEN")
            )
            # Merge LoRA weights into base model
            self.model = self.model.merge_and_unload()
        except:
            # Fallback: load as standard causal LM
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                token=os.getenv("HF_TOKEN")
            )
        
        self.model.eval()

    def predict(
        self,
        prompt: str = Input(description="Input text prompt"),
        max_length: int = Input(description="Maximum length of generated text", default=512),
        temperature: float = Input(description="Sampling temperature (0.0-2.0)", default=0.7, ge=0.0, le=2.0),
        top_p: float = Input(description="Nucleus sampling parameter", default=0.95, ge=0.0, le=1.0),
        top_k: int = Input(description="Top-k sampling", default=50, ge=0),
    ) -> str:
        """Generate text based on input prompt"""
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate with specified parameters
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )
        
        return generated_text
