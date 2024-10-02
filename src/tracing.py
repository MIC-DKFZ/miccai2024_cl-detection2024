# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: Apache-2.0

import torch
from utils.model import load_model

filepath = "/path/to/model/final_model.pt"

# Assuming you have a pre-trained PyTorch model
model = load_model(model_name="UNet", encoder_name="resnet18")
model.load_state_dict(torch.load(filepath))

# Put the model in evaluation mode
model.eval()
torch.save(model, filepath[:-3] + "_full_model.pt")

# Create a sample input tensor with the same size as your image
example_input = torch.randn(1, 1, 1024, 1024)  # Change image_height, image_width as needed

# Use tracing to convert the model to TorchScript
traced_model = torch.jit.trace(model, example_input)

# Save the TorchScript model for deployment
traced_model.save(filepath[:-3] + "_traced.pt")
