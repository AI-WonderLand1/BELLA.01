"""PyTorch Math Model for solving mathematical problems."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple


class MathProblemType:
    """Types of math problems the model can handle."""
    ARITHMETIC = "arithmetic"
    ALGEBRA = "algebra"
    CALCULUS = "calculus"
    STATISTICS = "statistics"
    GEOMETRY = "geometry"


class MathTokenizer:
    """Simple tokenizer for mathematical expressions."""
    
    # Special tokens
    PAD = 0
    SOS = 1  # Start of sequence
    EOS = 2  # End of sequence
    EQ = 3   # Equals sign
    X = 4    # Variable x
    PLUS = 5
    MINUS = 6
    MUL = 7
    DIV = 8
    POW = 9
    LPAREN = 10
    RPAREN = 11
    
    # Math-specific tokens
    SIN = 12
    COS = 13
    TAN = 14
    SQRT = 15
    INTEGRAL = 16
    LIMIT = 17
    
    # Numbers 0-9 as separate tokens or embedded
    NUM_START = 18
    
    VOCAB_SIZE = 100  # Enough for math tokens + numbers


class MathEmbedding(nn.Module):
    """Embedding layer for math expressions."""
    
    def __init__(self, vocab_size: int = MathTokenizer.VOCAB_SIZE, d_model: int = 128):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = None  # Will be set based on sequence length
    
    def forward(self, x: torch.Tensor, max_seq_len: int = 128) -> torch.Tensor:
        """Forward pass with positional encoding."""
        batch_size, seq_len = x.shape
        
        # Add positional encoding
        position = torch.arange(seq_len, device=x.device).unsqueeze(0).float()
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=x.device).float() * 
                           (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(1, seq_len, self.d_model, device=x.device)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x + pe
        return x


class MathProblemEncoder(nn.Module):
    """Encoder for math problems, supporting various problem types."""
    
    def __init__(self, d_model: int = 128, n_head: int = 4, n_layer: int = 2):
        super().__init__()
        self.d_model = d_model
        
        # Multi-head attention for math context
        self.attention = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        
        # Position-wise feedforward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.n_layer = n_layer
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode math problem."""
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feedforward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class MathSolverModel(nn.Module):
    """Neural solver for mathematical problems."""
    
    def __init__(self,
                 vocab_size: int = MathTokenizer.VOCAB_SIZE,
                 d_model: int = 256,
                 n_head: int = 4,
                 n_layer: int = 3,
                 max_seq_len: int = 256):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.embedding = MathEmbedding(vocab_size, d_model)
        
        # Problem encoder
        self.encoder = MathProblemEncoder(d_model, n_head, n_layer)
        
        # Decoder for generating math solutions
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_head,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=n_layer)
        
        # Output vocabulary projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        self.apply(self._weights_init)
    
    def _weights_init(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self,
                src: torch.Tensor,        # Input math expression (batch, seq_len)
                tgt: Optional[torch.Tensor] = None,  # Target solution (batch, seq_len)
                problem_type: MathProblemType = MathProblemType.ARITHMETIC,
                teacher_forcing_ratio: float = 0.5) -> Dict[str, torch.Tensor]:
        """Forward pass for math solving."""
        
        # Embed input
        src_embedded = self.embedding(src)
        src_encoded = self.encoder(src_embedded)
        
        if tgt is not None:
            # Training mode with teacher forcing
            tgt_embedded = self.embedding(tgt)
            
            # Causal mask for decoder
            seq_len = tgt.size(1)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            
            # Decoder
            tgt_output = self.decoder(tgt_embedded, src_encoded, tgt_mask=mask)
            logits = self.output_proj(tgt_output)
            
            # Compute loss
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_tgt = tgt[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=MathTokenizer.PAD)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_tgt.view(-1))
            
            return {"logits": logits, "loss": loss}
        
        else:
            # Inference mode: autoregressive generation
            return self._generate(src)
    
    def _generate(self, src: torch.Tensor, max_len: int = 128) -> Dict[str, torch.Tensor]:
        """Autoregressive generation of math solution."""
        src_embedded = self.embedding(src)
        src_encoded = self.encoder(src_embedded)
        
        # Start with SOS token
        generated = [MathTokenizer.SOS]
        
        for _ in range(max_len):
            # Embed generated tokens so far
            gen_tensor = torch.tensor([generated], dtype=torch.long, device=src.device)
            gen_embedded = self.embedding(gen_tensor)
            
            # Add positional
            pos = len(generated)
            pos_encoding_vec = torch.zeros(1, 1, self.d_model, device=src.device)
            # Simplified positional encoding
            div_term = torch.exp(torch.arange(0, self.d_model, 2, device=src.device).float() * 
                              (-math.log(10000.0) / self.d_model))
            pos_encoding_vec[0, 0, 0::2] = torch.sin(torch.tensor(pos, device=src.device) * div_term)
            pos_encoding_vec[0, 0, 1::2] = torch.cos(torch.tensor(pos, device=src.device) * div_term)
            gen_embedded = gen_embedded + pos_encoding_vec
            
            # Decoder with encoder states
            seq_len = len(generated)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=src.device), diagonal=1).bool()
            
            dec_out = self.decoder(gen_embedded.transpose(0, 1), src_encoded.transpose(0, 1), tgt_mask=mask.transpose(0, 1))
            logits = self.output_proj(dec_out.transpose(0, 1))
            
            # Get last token logits
            last_logit = logits[0, -1, :]
            next_token = torch.argmax(last_logit).item()
            
            generated.append(next_token)
            
            # Stop at EOS
            if next_token == MathTokenizer.EOS:
                break
        
        # Return as tensor
        gen_tensor = torch.tensor([generated], dtype=torch.long, device=src.device)
        return {"logits": gen_tensor, "solution_tokens": torch.tensor([generated], dtype=torch.long)}


def solve_arithmetic(expression: str, model: MathSolverModel = None) -> Dict[str, Any]:
    """Solve a simple arithmetic problem."""
    # Tokenize simple arithmetic
    # This is a simplified version - full implementation would need proper tokenization
    print(f"Solving: {expression}")
    if model:
        # Would use model to solve
        pass
    
    # Simple fallback: evaluate safely
    try:
        # Only allow basic arithmetic
        import re
        if re.match(r'^[\d+\-*/().\s]+$', expression):
            result = eval(expression)
            return {"success": True, "expression": expression, "result": result}
    except:
        pass
    
    return {"success": False, "expression": expression, "result": None}


def generate_math_solution(problem: str, model: MathSolverModel = None) -> str:
    """Generate a step-by-step math solution."""
    if model:
        # Use model to generate solution
        pass
    
    # Simple heuristic-based solution generation
    problem_lower = problem.lower()
    
    if '+' in problem or 'add' in problem_lower:
        # Extract numbers
        numbers = [int(s) for s in problem.split() if s.isdigit()]
        if len(numbers) >= 2:
            return f"To solve {problem}:\n1. Add the numbers: {numbers[0]} + {numbers[1]} = {numbers[0] + numbers[1]}\n2. Result: {numbers[0] + numbers[1]}"
    
    if 'x' in problem or 'solve' in problem_lower:
        return f"To solve {problem}:\n1. Isolate the variable x\n2. Apply inverse operations\n3. Verify solution"
    
    return f"Math problem: {problem}\nSolution: See step-by-step analysis."


# Test function
if __name__ == "__main__":
    print("Math model module loaded successfully")
    print(f"Tokenizer vocab size: {MathTokenizer.VOCAB_SIZE}")
    
    # Quick test
    result = solve_arithmetic("15 + 27")
    print(f"Arithmetic test: {result}")
    
    result = solve_arithmetic("12 * 8")
    print(f"Multiplication test: {result}")
