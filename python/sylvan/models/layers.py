"""Mixture of Experts layers for the Sylvan project."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseMoE(nn.Module):
    """
    A Sparse Mixture of Experts layer.
    Routes the input to the top-k experts based on a gating network.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.input_dim = input_dim

        self.gate = nn.Linear(input_dim, num_experts)
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, output_dim)
            ) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x_flat = x.view(-1, self.input_dim) # [B*T, D]
        
        logits = self.gate(x_flat) # [B*T, num_experts]
        probs = F.softmax(logits, dim=-1) # [B*T, num_experts]
        
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1) # [B*T, top_k]
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True) # Normalize top-k probabilities

        output = torch.zeros(x_flat.shape[0], self.experts[0][2].out_features, device=x.device, dtype=x.dtype)
        
        for k in range(self.top_k):
            expert_indices = top_k_indices[:, k] # [B*T]
            expert_probs = top_k_probs[:, k].unsqueeze(-1) # [B*T, 1]
            
            for i, expert in enumerate(self.experts):
                mask = expert_indices == i
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = expert(expert_input)
                    output[mask] += expert_probs[mask] * expert_output
                    
        return output.view(batch_size, seq_len, -1)
