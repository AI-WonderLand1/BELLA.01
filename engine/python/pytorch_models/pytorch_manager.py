"""Unified PyTorch Manager for BELLA.01 Hybrid AI Framework.

Combines code generation, text generation, and regression models
with HuggingFace and local model support.
"""
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Try to import huggingface_hub, fall back gracefully
try:
    from huggingface_hub import hf_hub_download, list_repo_tags, login
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("⚠️ huggingface_hub not installed. Install with: pip install huggingface_hub")

from .codegen_model import CodegenConfig, HybridCodegenModel, load_model as load_codegen, save_model as save_codegen
from .regression_model import RegressionModel, train_regression_model, predict_regression


class PyTorchTask:
    """Enum-like task types for the hybrid model."""
    CODE_GENERATION = "code_generation"
    TEXT_GENERATION = "text_generation"
    REGRESSION = "regression"
    HF_DOWNLOAD = "hf_download"
    LOCAL_LOAD = "local_load"


class ModelInfo:
    """Metadata about a loaded model."""
    def __init__(self, name: str, model_type: str, model: Any, config: Any,
                 source: str = "local", path: str = "", hf_repo: str = None):
        self.name = name
        self.model_type = model_type
        self.model = model
        self.config = config
        self.source = source  # "local" or "hf"
        self.path = path
        self.hf_repo = hf_repo


class PyTorchManager:
    """Manager for hybrid PyTorch models with HF and local support."""

    def __init__(self,
                 model_dir: str = "/home/bobbi/BELLA.01/engine/python/pytorch_models",
                 device: str = "auto",
                 hf_token: str = None):
        self.model_dir = Path(model_dir)
        self.hf_token = hf_token or os.environ.get("HF_TOKEN", "")

        # Auto-detect device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device

        self.models: Dict[str, ModelInfo] = {}
        self.configs: Dict[str, CodegenConfig] = {}

        # Ensure model directory exists
        self.model_dir.mkdir(parents=True, exist_ok=True)

        print(f"🔧 PyTorchManager initialized on device: {self.device}")
        print(f"   HuggingFace token: {'configured' if self.hf_token else 'not set'}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        print(f"   MPS available: {torch.backends.mps.is_available()}")

    def _ensure_device(self, tensor_or_model) -> torch.device:
        """Ensure tensor/model is on correct device."""
        if isinstance(tensor_or_model, torch.Tensor):
            return tensor_or_model.to(self.device)
        if isinstance(tensor_or_model, nn.Module):
            return self._ensure_device(tensor_or_model)
        return self.device

    def load_local_model(self,
                         name: str,
                         model_type: str,
                         model_path: str = None,
                         config: CodegenConfig = None) -> Optional[ModelInfo]:
        """Load a model from local storage."""
        model_path = Path(model_path) if model_path else self.model_dir / f"{name}.pth"

        if model_type == "codegen":
            if model_path.exists() and model_path.is_file():
                model = HybridCodegenModel(config or CodegenConfig())
                state = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state)
                model = model.to(self.device)
                model_info = ModelInfo(name, model_type, model, config, source="local", path=str(model_path))
                self.models[name] = model_info
                print(f"📁 Loaded local codegen model: {name} from {model_path}")
                return model_info
            else:
                print(f"❌ Local model not found: {model_path}")
                return None

        elif model_type == "regression":
            model = RegressionModel(input_dim=config.d_model if config else 256)
            if model_path.exists():
                state = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state)
            model = model.to(self.device)
            model_info = ModelInfo(name, model_type, model, config, source="local", path=str(model_path))
            self.models[name] = model_info
            print(f"📁 Loaded local regression model: {name}")
            return model_info

        return None

    def load_hf_model(self,
                      name: str,
                      model_type: str,
                      repo_id: str,
                      revision: str = "main") -> Optional[ModelInfo]:
        """Load a model from HuggingFace Hub."""
        if not HF_HUB_AVAILABLE:
            print("❌ huggingface_hub not available. Install with: pip install huggingface_hub")
            return None

        try:
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=f"{name}.pth" if model_type == "codegen" else "pytorch_model.bin",
                revision=revision,
                token=self.hf_token,
            )

            model_path = Path(model_path)

            if model_type == "codegen":
                model = HybridCodegenModel(CodegenConfig())
                state = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state)
                model = model.to(self.device)
                model_info = ModelInfo(name, model_type, model, CodegenConfig(),
                                      source="hf", path=str(model_path), hf_repo=repo_id)
                self.models[name] = model_info
                print(f"🤖 Loaded HF codegen model: {name} from {repo_id}")
                return model_info

            elif model_type == "regression":
                model = RegressionModel(input_dim=256)
                state = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state)
                model = model.to(self.device)
                model_info = ModelInfo(name, model_type, model, CodegenConfig(),
                                      source="hf", path=str(model_path), hf_repo=repo_id)
                self.models[name] = model_info
                print(f"🤖 Loaded HF regression model: {name} from {repo_id}")
                return model_info

        except Exception as e:
            print(f"❌ Failed to load HF model {repo_id}: {e}")
            return None

    def init_codegen_model(self,
                          name: str = "codegen_default",
                          config: CodegenConfig = None,
                          source: str = "local",
                          hf_repo: str = None,
                          hf_revision: str = "main") -> Optional[ModelInfo]:
        """Initialize or load a code generation model (local or HF)."""
        if source == "hf" and hf_repo:
            return self.load_hf_model(name, "codegen", hf_repo, hf_revision)
        else:
            return self.load_local_model(name, "codegen",
                                         self.model_dir / f"{name}.pth", config)

    def init_regression_model(self,
                              name: str = "regression_default",
                              input_dim: int = 256,
                              hidden_dims: list = None,
                              source: str = "local",
                              hf_repo: str = None) -> Optional[ModelInfo]:
        """Initialize regression model (local or HF)."""
        if source == "hf" and hf_repo:
            return self.load_hf_model(name, "regression", hf_repo)
        else:
            model = RegressionModel(input_dim=input_dim, hidden_dims=hidden_dims)
            model = model.to(self.device)
            model_info = ModelInfo(name, "regression", model, CodegenConfig(),
                                  source="local", path="")
            self.models[name] = model_info
            print(f"✅ Initialized local regression model: {name}")
            return model_info

    def generate_code(self,
                      model_info: ModelInfo,
                      prompt: str,
                      max_new_tokens: int = 128,
                      temperature: float = 0.8,
                      top_k: int = 50) -> str:
        """Generate code using a loaded model."""
        model = model_info.model
        model.eval()

        # Simple encoding - prompt to IDs
        prompt_ids = [ord(c) % 10000 for c in prompt][:64]
        input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(self.device)

        with torch.no_grad():
            # Autoregressive generation
            for _ in range(max_new_tokens):
                crop_start = max(0, input_ids.size(1) - model.config.max_seq_len)
                input_crop = input_ids[:, crop_start:]

                # Causal mask
                seq_len = input_crop.size(1)
                mask = torch.triu(torch.ones(seq_len, seq_len, device=input_crop.device),
                                  diagonal=1).bool()

                # Forward through our model components
                embedded = model.embedding(input_crop)
                pos_encoded = model.pos_encoding(embedded)
                decoder_output = model.transformer_decoder(pos_encoded, pos_encoded, tgt_mask=mask)
                logits = model.code_head(decoder_output)

                last_logits = logits[:, -1, :] / temperature
                probs = F.softmax(last_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                input_ids = torch.cat([input_ids, next_token], dim=1)

                # Stop if EOS-like token
                if next_token.item() == 2:
                    break

        # Decode
        generated_chars = []
        for i in range(1, len(input_ids[0])):
            idx = input_ids[0][i].item()
            if 32 <= idx <= 126:  # Printable ASCII
                generated_chars.append(chr(idx))

        return prompt + "".join(generated_chars)

    def train_codegen(self,
                      model_info: ModelInfo,
                      train_data: List[Dict[str, torch.Tensor]],
                      epochs: int = 3,
                      lr: float = 1e-4) -> Dict[str, float]:
        """Train code generation model."""
        model = model_info.model
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        num_batches = 0

        for epoch in range(epochs):
            for batch in train_data:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                output = model(input_ids=input_ids, task="code", labels=labels)

                loss = output.get("loss", criterion(
                    output["logits"][:, :-1, :].contiguous().view(-1, output["logits"].size(-1)),
                    labels[:, 1:].contiguous().view(-1)
                ))

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        print(f"✅ Codegen training complete - Avg loss: {avg_loss:.4f}")
        return {"average_loss": avg_loss, "epochs": epochs, "batches": num_batches}

    def predict_regression(self,
                          model_info: ModelInfo,
                          x: torch.Tensor) -> torch.Tensor:
        """Make regression predictions."""
        model = model_info.model
        model.eval()
        with torch.no_grad():
            x_device = x.to(self.device)
            outputs = model(x_device)
        return outputs.cpu()

    def save_model(self, name: str) -> Optional[str]:
        """Save a model to local storage."""
        if name in self.models:
            model_info = self.models[name]
            model_path = self.model_dir / f"{name}.pth"
            torch.save(model_info.model.state_dict(), model_path)
            model_info.path = str(model_path)
            print(f"💾 Saved model: {name} to {model_path}")
            return str(model_path)
        return None

    def save_all(self) -> Dict[str, str]:
        """Save all models."""
        saved = {}
        for name in list(self.models.keys()):
            path = self.save_model(name)
            if path:
                saved[name] = path
        return saved

    def get_model_info(self) -> Dict[str, Dict]:
        """Get info about all models."""
        info = {}
        for name, model_info in self.models.items():
            model = model_info.model
            param_count = sum(p.numel() for p in model.parameters()) if hasattr(model, 'parameters') else 0
            info[name] = {
                "type": model_info.model_type,
                "source": model_info.source,
                "parameters": param_count,
                "device": str(next(model.parameters()).device) if hasattr(model, 'parameters') else "unknown",
                "parameters_path": model_info.path,
                "hf_repo": model_info.hf_repo,
            }
        return info


def create_manager(model_dir: str = None, device: str = "auto", hf_token: str = None) -> PyTorchManager:
    """Create a PyTorchManager for BELLA.01 hybrid AI framework."""
    return PyTorchManager(model_dir=model_dir or "/home/bobbi/BELLA.01/engine/python/pytorch_models",
                         device=device, hf_token=hf_token)
