"""PyTorch Code Generation Model for LLM-style code completion."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class CodegenConfig:
    """Configuration for code generation model."""
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 256,
        n_head: int = 4,
        n_layer: int = 2,
        max_seq_len: int = 512,
        d_ff: int = 512,
        dropout: float = 0.1,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_head = n_head
        self.n_layer = n_layer
        self.max_seq_len = max_seq_len
        self.d_ff = d_ff
        self.dropout = dropout


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model)
        self.register_buffer("pe", pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, d_model)
        x = x + (self.pe[:, :x.size(1), :]).requires_grad_(False)
        return self.dropout(x)


class HybridCodegenModel(nn.Module):
    """Hybrid model combining pre-defined structure with custom layers for code/text generation."""
    
    def __init__(self, config: CodegenConfig):
        super().__init__()
        self.config = config
        
        # --- PRE-DEFINED BACKBONE: Transformer decoder layers ---
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_len, config.dropout)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_head,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.n_layer)
        
        # --- CUSTOM HEAD: Task-specific output layers ---
        # For multi-task: code generation + text + regression
        self.code_head = nn.Linear(config.d_model, config.vocab_size)  # vocabulary prediction
        self.text_head = nn.Linear(config.d_model, config.vocab_size)  # text generation
        self.regression_head = nn.Linear(config.d_model, 1)  # regression output
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Standard weight initialization."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        task: str = "code",  # "code", "text", or "regression"
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass supporting multiple tasks.
        
        Args:
            input_ids: (batch, seq_len) tensor of token IDs
            task: "code", "text", or "regression"
            labels: Optional labels for computing loss
        
        Returns:
            Dict with logits and optionally loss
        """
        # Embed and add positional encoding
        x = self.embedding(input_ids)  # (batch, seq_len, d_model)
        x = self.pos_encoding(x)
        
        # Create causal mask for decoder
        seq_len = input_ids.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device), diagonal=1).bool()
        
        # Transformer decoder (no source masking since we only have target)
        x = self.transformer_decoder(x, x, tgt_mask=mask)
        
        # Task-specific heads
        logits = None
        loss = None
        
        if task == "code":
            logits = self.code_head(x)  # (batch, seq_len, vocab_size)
            if labels is not None:
                # Shift logits and labels for next-token prediction
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        elif task == "text":
            logits = self.text_head(x)
            if labels is not None:
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        elif task == "regression":
            # Global pooling + regression head
            pooled = x.mean(dim=1)  # (batch, d_model)
            logits = self.regression_head(pooled)  # (batch, 1)
            if labels is not None:
                loss = F.mse_loss(logits.squeeze(), labels.squeeze())
        
        result = {"logits": logits}
        if loss is not None:
            result["loss"] = loss
        return result
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        task: str = "code",
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
    ) -> torch.Tensor:
        """Generate new tokens autoregressively."""
        self.eval()
        
        for _ in range(max_new_tokens):
            # Crop input to max_seq_len
            crop_start = max(0, input_ids.size(1) - self.config.max_seq_len)
            input_crop = input_ids[:, crop_start:]
            
            # Forward pass
            output = self.forward(input_ids=input_crop, task=task)
            logits = output["logits"]
            
            if logits is None:
                break
            
            # Get last token logits
            last_logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k is not None:
                # Remove tokens with probability < top_k
                topk_values, _ = torch.topk(last_logits, min(top_k, last_logits.size(-1)))
                indices_to_remove = last_logits < topk_values[:, [-1]]
                last_logits[indices_to_remove] = -float("inf")
            
            # Sample next token
            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop if EOS token (assuming 2 is EOS, adjust as needed)
            if next_token.item() == 2:  # EOS token
                break
        
        return input_ids


def load_model(model_path: str, config: Optional[CodegenConfig] = None) -> HybridCodegenModel:
    """Load a saved PyTorch model."""
    if config is None:
        config = CodegenConfig()
    
    model = HybridCodegenModel(config)
    
    if model_path and model_path.exists():
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
        print(f"✅ Model loaded from {model_path}")
    
    return model


def save_model(model: HybridCodegenModel, model_path: str) -> None:
    """Save PyTorch model state."""
    torch.save(model.state_dict(), model_path)
    print(f"💾 Model saved to {model_path}")
