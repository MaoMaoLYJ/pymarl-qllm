import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class QLLMMixer(nn.Module):
    def __init__(self):
        super(QLLMMixer, self).__init__()
    def forward(self, agents_q, global_state):
        a = agents_q.shape[0]
        b = agents_q.shape[1]
        agents_q = agents_q.reshape(-1, agents_q.shape[-1])
        global_state = global_state.reshape(-1, global_state.shape[-1])
        batch_size = agents_q.shape[0]
        device = global_state.device
        
        # Extract agent features (2 agents, 5 features each)
        agent1_feat = global_state[:, 0:5]   # [health, cooldown, x, y, shield]
        agent2_feat = global_state[:, 5:10]
        
        # Extract enemy features (64 enemies, 3 features each)
        enemy_feat = global_state[:, 10:202].view(batch_size, 64, 3)
        
        # Compute agent states
        agent1_health = agent1_feat[:, 0]
        agent2_health = agent2_feat[:, 0]
        agent1_cooldown = agent1_feat[:, 1]
        agent2_cooldown = agent2_feat[:, 1]
        agent1_shield = agent1_feat[:, 4]
        agent2_shield = agent2_feat[:, 4]
        
        # Mask for dead agents
        agent1_dead = (agent1_health == 0.0).float()
        agent2_dead = (agent2_health == 0.0).float()
        
        # Survivability factor (health + shield, scaled to [0, 2])
        agent1_survival = agent1_health + agent1_shield
        agent2_survival = agent2_health + agent2_shield
        
        # Cooldown factor (ready to attack gets higher weight)
        agent1_cd_factor = 1.0 / (1.0 + agent1_cooldown)
        agent2_cd_factor = 1.0 / (1.0 + agent2_cooldown)
        
        # Compute threat by distance to enemies
        agent1_pos = agent1_feat[:, 2:4].unsqueeze(1)  # [batch, 1, 2]
        agent2_pos = agent2_feat[:, 2:4].unsqueeze(1)  # [batch, 1, 2]
        enemy_pos = enemy_feat[:, :, 1:3]  # [batch, 64, 2]
        
        # Compute distances to all enemies
        agent1_dist = torch.norm(agent1_pos - enemy_pos, dim=2, p=2)
        agent2_dist = torch.norm(agent2_pos - enemy_pos, dim=2, p=2)
        
        # Zergling attack range is ~0.01 in normalized coordinates
        # Colossus attack range is 0.25, so within 0.1 is high threat
        threat_threshold = 0.1
        agent1_threat = (agent1_dist < threat_threshold).float().sum(dim=1)
        agent2_threat = (agent2_dist < threat_threshold).float().sum(dim=1)
        
        # Normalize threat by max possible enemies (64)
        agent1_threat_norm = agent1_threat / 64.0
        agent2_threat_norm = agent2_threat / 64.0
        
        # Combine factors with careful scaling
        # Base weights consider survival, cooldown readiness, and threat
        agent1_base = (agent1_survival * agent1_cd_factor * (1.0 + 2.0 * agent1_threat_norm))
        agent2_base = (agent2_survival * agent2_cd_factor * (1.0 + 2.0 * agent2_threat_norm))
        
        # Apply dead agent mask
        agent1_weight = agent1_base * (1.0 - agent1_dead)
        agent2_weight = agent2_base * (1.0 - agent2_dead)
        
        # Stack weights and apply softmax for proper distribution
        weights = torch.stack([agent1_weight, agent2_weight], dim=1)
        weights = torch.softmax(weights, dim=1)
        
        # Compute weighted sum of Q-values
        global_q = (weights * agents_q).sum(dim=1, keepdim=True)
        
        # Add bias based on overall battle state
        # Use enemy health as indicator of remaining threat
        enemy_health = enemy_feat[:, :, 0]  # [batch, 64]
        avg_enemy_health = enemy_health.mean(dim=1, keepdim=True)  # [batch, 1]
        bias = -0.5 * avg_enemy_health  # Negative bias as enemies are alive
        
        global_q = global_q + bias
        
        global_q=global_q
        return (global_q*agents_q.shape[-1]).reshape(a, b, 1).cuda()