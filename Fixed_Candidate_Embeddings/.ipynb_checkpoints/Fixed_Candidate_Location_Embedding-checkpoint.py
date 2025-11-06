#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import torch.nn as nn

class Candidate_Location_Coordinate:
    def __init__(self, grid_size=16, image_size=256):
        """
        Initialize the BoundaryMaskCreator.

        Parameters:
        - grid_size: The size of the grid for boundary point extraction
        - image_size: The dimensions of the image (assumed square)
        """
        self.grid_size = grid_size
        self.image_size = image_size
        self.gap = image_size // grid_size
        self.boundary_points = self._generate_candidate_location()
        self.boundary_points_tensor = torch.tensor(self.boundary_points, dtype=torch.float32)

    def _generate_candidate_location(self):
        """
        Generate candidate location points.
        
        Returns:
        - boundary_points: Array of boundary points
        """
        boundary_points = []
        for i in range(self.grid_size):
            # Horizontal lines
            y = i * self.gap
            for x in range(0, self.image_size, self.gap):
                boundary_points.append((x, y))
            
            # Vertical lines
            x = i * self.gap
            for y in range(0, self.image_size, self.gap):
                boundary_points.append((x, y))
        
        boundary_points = np.array(boundary_points)
        boundary_points = np.unique(boundary_points, axis=0)
        return boundary_points

class Candidate_location_prompt_embedding(nn.Module):
    def __init__(self, embed_dim=384, num_pos_feats=128, input_image_size=(256, 256), num_boxes=2):
        """
        A simpler version of PromptEncoder for encoding bounding box coordinates.

        Arguments:
        embed_dim -- Dimension of the embedding (e.g., 384)
        num_pos_feats -- Number of positional features (e.g., 128)
        input_image_size -- Size of the input image (height, width)
        num_boxes -- Number of key points used for encoding (default: 2 for top-left and bottom-right corners)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.num_boxes = num_boxes  # Usually 2 for two corners

        # Positional embedding matrix
        self.register_buffer("positional_embedding", torch.randn((2, num_pos_feats)) * embed_dim // 2)

        # Learnable embeddings for each box corner
        self.box_embeddings = nn.ModuleList([nn.Embedding(1, num_pos_feats * 2) for _ in range(num_boxes)])

    def forward(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Embeds bounding box coordinates.

        Arguments:
        boxes -- Tensor of shape (batch_size, num_boxes, 2) containing (x, y) coordinates.

        Returns:
        Tensor of shape (batch_size, num_boxes, embed_dim).
        """
        batch_size = boxes.shape[0]

        # Compute positional embeddings
        box_embedding = self.compute_positional_embedding(boxes)

        # Add learnable embeddings
        for i in range(self.num_boxes):
            box_embedding[:, i, :] += self.box_embeddings[i].weight

        return box_embedding.view(batch_size, -1, box_embedding.shape[-1])  # Reshape to match expected output

    def compute_positional_embedding(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute positional embedding for input coordinates.
    
        Arguments:
        coords -- Tensor of shape (batch_size, num_boxes, 2)
    
        Returns:
        Tensor with encoded positional information.
        """
        coords = coords.clone().to(torch.float32)  # Convert to float before division
    
        # Normalize coordinates to [0, 1] range
        height, width = self.input_image_size
        coords[:, :, 0] /= width
        coords[:, :, 1] /= height
    
        # Scale to [-1, 1] range
        coords = 2 * coords - 1
        coords = coords @ self.positional_embedding  # Apply embedding matrix
    
        # Convert to sinusoidal embeddings
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)  # Concatenate sin and cos

# Create candidate locations
creator = Candidate_Location_Coordinate(grid_size=32)
Candidate_Location = creator._generate_candidate_location()
print(f"Candidate_Location {Candidate_Location.shape}")

# Convert to tensor and prepare input boxes
Candidate_Location = torch.tensor(Candidate_Location, dtype=torch.float32)
print("Candidate_Location", Candidate_Location.shape)

input_boxes = torch.zeros((Candidate_Location.size(0), 4), dtype=Candidate_Location.dtype)

# Fill the new tensor
input_boxes[:, 0] = Candidate_Location[:, 0]  # First column
input_boxes[:, 1] = Candidate_Location[:, 1]  # Second column
input_boxes[:, 2] = Candidate_Location[:, 0]  # Repeat first column
input_boxes[:, 3] = Candidate_Location[:, 1]  # Repeat second column

coords = input_boxes.reshape(-1, 2, 2)
print("coords", coords.shape)

# Generate embeddings
Candidate_embedding = Candidate_location_prompt_embedding(embed_dim=256, input_image_size=(256, 256))
Candidate_embeddings = Candidate_embedding(boxes=coords)
print("Candidate Location Prompt Embeddings shape:", Candidate_embeddings.shape)

Candidate_emb = Candidate_embeddings[:, 0, :]
print("Candidate_emb:::::", Candidate_emb.shape)

# Save embeddings
i = 1
torch.save(Candidate_emb, './Candidate_Prompt_Embedding' + str(i) + '.pt')