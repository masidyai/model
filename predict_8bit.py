import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM
from cog import BasePredictor, Input
import os

class Predictor(BasePredictor):
    def setup(self):
        """Load model and tokenizer on startup"""
        model_name = "masidyai/qwen3.5-9b-masidy"
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=os.getenv("HF_TOKEN")
        )
        
        # Load model with 8-bit quantization for memory efficiency
        try:
            self.model = AutoPeftModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map="auto",
                trust_remote_code=True,
                token=os.getenv("HF_TOKEN")
            )
            self.model = self.model.merge_and_unload()
        except:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map="auto",
                trust_remote_code=True,
                token=os.getenv("HF_TOKEN")
            )
        
        self.model.eval()

    def predict(
        self,
        prompt: str = Input(description="Input text prompt"),
        max_length: int = Input(description="Maximum length of generated text", default=256),
        temperature: float = Input(description="Sampling temperature (0.0-2.0)", default=0.7, ge=0.0, le=2.0),
        top_p: float = Input(description="Nucleus sampling parameter", default=0.95, ge=0.0, le=1.0),
        top_k: int = Input(description="Top-k sampling", default=50, ge=0),
    ) -> str:
        """Generate text based on input prompt"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
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
        
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )
        
        return generated_text
