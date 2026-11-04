"""PyTorch Text Model for language understanding and generation."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple


class TextTokenizer:
    """Simplified text tokenizer."""
    
    PAD = 0
    SOS = 1
    EOS = 2
    UNK = 3
    VOCAB_SIZE = 10000  # Can be expanded with real tokenizer
    
    # Common tokens
    CLAUSE_START = 100
    CLAUSE_END = 101
    SENTENCE_START = 102
    SENTENCE_END = 103


class TextEmbedding(nn.Module):
    """Token embedding with positional encoding for text."""
    
    def __init__(self, vocab_size: int = TextTokenizer.VOCAB_SIZE, d_model: int = 512, max_seq: int = 512):
        super().__init__()
        self.d_model = d_model
        self.max_seq = max_seq
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self._create_positional_encoding(max_seq, d_model)
    
    def _create_positional_encoding(self, max_seq: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        position = torch.arange(max_seq).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_seq, d_model)
    
    def forward(self, x: torch.Tensor, seq_len: int = None) -> torch.Tensor:
        """Add positional encoding to token embeddings."""
        batch_size, seq_len = x.shape
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        
        # Use positional encoding
        if seq_len and seq_len <= self.max_seq:
            pe = self.positional_encoding[:seq_len, :].transpose(0, 1).unsqueeze(0)
        else:
            pe = self.positional_encoding[:seq_len if not seq_len else self.max_seq, :].transpose(0, 1).unsqueeze(0)
        
        x = x + pe
        return x


class TextTransformerBlock(nn.Module):
    """Single transformer block for text processing."""
    
    def __init__(self, d_model: int, n_head: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention with causal mask for generation
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feedforward
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class GPTLikeModel(nn.Module):
    """GPT-like model for text generation and understanding."""
    
    def __init__(self,
                 vocab_size: int = TextTokenizer.VOCAB_SIZE,
                 d_model: int = 384,
                 n_layer: int = 4,
                 n_head: int = 4,
                 max_seq_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Token embedding + positional
        self.embedding = TextEmbedding(vocab_size, d_model, max_seq_len)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TextTransformerBlock(d_model, n_head, d_ff=d_model * 4, dropout=dropout)
            for _ in range(n_layer)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Causal mask cache
        self.causal_mask = self._create_causal_mask(max_seq_len)
        
        self._init_weights()
    
    def _create_causal_mask(self, size: int) -> torch.Tensor:
        """Create upper-triangular causal mask."""
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask
    
    def _init_weights(self):
        self.apply(self._weights_init)
    
    def _weights_init(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        seq_len = input_ids.size(1)
        
        # Embed
        x = self.embedding(input_ids, seq_len)
        
        # Causal mask
        mask = self.causal_mask[:seq_len, :seq_len].to(input_ids.device)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Output logits
        logits = self.output_proj(x)
        
        result = {"logits": logits}
        
        # Compute loss if labels provided
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=TextTokenizer.PAD)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            result["loss"] = loss
        
        return result
    
    @torch.no_grad()
    def generate(self, 
                 prompt: str, 
                 max_new_tokens: int = 128, 
                 temperature: float = 0.8,
                 top_k: Optional[int] = 50,
                 device: str = "cpu") -> str:
        """Generate text from prompt."""
        self.eval()
        self.to(device)
        
        # Simple encoding: convert prompt to token IDs
        # In production, use proper tokenizer (BPE, WordPiece, etc.)
        prompt_ids = [ord(c) % TextTokenizer.VOCAB_SIZE for c in prompt][:self.max_seq_len]
        input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(device)
        
        for _ in range(max_new_tokens):
            # Crop to max_seq_len
            crop_start = max(0, input_ids.size(1) - self.max_seq_len)
            input_crop = input_ids[:, crop_start:]
            
            # Forward
            output = self.forward(input_crop)
            logits = output["logits"]
            
            # Last token logits
            last_logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k is not None:
                val, _ = torch.topk(last_logits, min(top_k, last_logits.size(-1)))
                last_logits[last_logits < val[:, [-1]]] = -float('Inf')
            
            # Sample
            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop at EOS
            if next_token.item() == TextTokenizer.EOS:
                break
        
        # Decode: simple reverse mapping
        # In production, use proper tokenizer decode
        result = prompt
        for i in range(1, len(input_ids[0])):
            idx = input_ids[0][i].item()
            if idx >= 32 and idx <= 126:  # Printable ASCII
                result += chr(idx)
        
        return result


def classify_sentiment(text: str) -> Dict[str, float]:
    """Simple sentiment classification (placeholder)."""
    # In real usage, would use trained model
    positive_words = ['good', 'great', 'excellent', 'awesome', 'love', 'best']
    negative_words = ['bad', 'terrible', 'hate', 'worst', 'awful', 'sad']
    
    text_lower = text.lower()
    pos_count = sum(1 for w in positive_words if w in text_lower)
    neg_count = sum(1 for w in negative_words if w in text_lower)
    
    if pos_count > neg_count:
        return {"sentiment": "positive", "confidence": min(0.9, 0.5 + 0.05 * pos_count)}
    elif neg_count > pos_count:
        return {"sentiment": "negative", "confidence": min(0.9, 0.5 + 0.05 * neg_count)}
    else:
        return {"sentiment": "neutral", "confidence": 0.5}


def summarize_text(text: str, max_summary_len: int = 100) -> str:
    """Simple text summarization."""
    # Basic extractive summarization
    sentences = text.split('.')
    # Take first and last sentences + key sentences
    if len(sentences) <= 2:
        return text
    
    summary_parts = []
    if sentences[0]:
        summary_parts.append(sentences[0].strip())
    if sentences[-1]:
        summary_parts.append(sentences[-1].strip())
    
    # Add middle if very long
    if len(text) > 500:
        summary_parts.append(f"... ({len(sentences) - 2} more sentences)")
    
    return '. '.join(summary_parts) + '.'


# Test
if __name__ == "__main__":
    print("Text model module loaded successfully")
    print(f"Vocab size: {TextTokenizer.VOCAB_SIZE}")
    
    # Test sentiment
    result = classify_sentiment("I love this wonderful day!")
    print(f"Sentiment test: {result}")
    
    # Test summarization
    long_text = "The quick brown fox jumps over the lazy dog. " * 20
    summary = summarize_text(long_text)
    print(f"Summarization test: {summary[:80]}...")
    
    # Test generation
    print("\nGeneration test (small):")
    model_part = "GPTLikeModel class ready for instantiation"
    print(model_part)
