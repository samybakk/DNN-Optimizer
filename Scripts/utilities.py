from torch import nn, optim, quantization
import torch

def get_pytorch_backbone_layers_id(model):
    # Find the first convolutional layer in the model
    for i, layer in enumerate(model.modules()):
        if isinstance(layer, nn.Conv2d):
            first_conv_layer_id = i
            break

    # Find the last convolutional layer in the model before the final pooling layer
    last_conv_layer_id = None
    for i, layer in reversed(list(enumerate(model.modules()))):
        if isinstance(layer, nn.Conv2d):
            last_conv_layer_id = i
            break
        if isinstance(layer, nn.AdaptiveAvgPool2d) or isinstance(layer, nn.AdaptiveMaxPool2d):
            break

    # Check that the model has a backbone network
    assert first_conv_layer_id is not None and last_conv_layer_id is not None

    # Get the IDs of the backbone layers
    backbone_layers = list(range(first_conv_layer_id, last_conv_layer_id + 1))

    return backbone_layers

def extract_backbone_layers(model, backbone_layer_ids):
    # Create a list of the modules to be included in the new model
    modules_to_include = [module for i, module in enumerate(model.modules()) if i in backbone_layer_ids]

    # Create a new model that includes only the selected modules
    new_model = nn.Sequential(*modules_to_include)

    # Create a list of the modules to be excluded from the new model
    modules_to_exclude = [module for i, module in enumerate(model.modules()) if i not in backbone_layer_ids]

    # Create a new model that includes only the excluded modules
    remaining_model = nn.Sequential(*modules_to_exclude)

    return new_model, remaining_model


def prune_backbone_with_gal(backbone, pruning_ratio,epochs,device):
    
    #input_size = 256
    # Define the generator and discriminator networks for the GAL technique
    generator = nn.Sequential(
        nn.Linear(backbone[0].in_channels, int(backbone[0].in_channels * 1.5)),
        nn.ReLU(),
        nn.Linear(int(backbone[0].in_channels * 1.5), backbone[0].in_channels)
    ).to(device=device)
    discriminator = nn.Sequential(
        nn.Linear(backbone[0].in_channels, int(backbone[0].in_channels * 0.75)),
        nn.ReLU(),
        nn.Linear(int(backbone[0].in_channels * 0.75), 1),
        nn.Sigmoid()
    ).to(device=device)

    # Define the optimizer for the generator and discriminator
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-3, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-3, betas=(0.5, 0.999))

    # Define the loss function for the generator and discriminator
    adversarial_loss = nn.BCELoss()

    

    # Create an empty tensor to hold the weights
    weight_tensor = None

    # Iterate over the layers in the backbone and concatenate their weights
    for layer in backbone:
        if hasattr(layer, 'weight'):
            weights = layer.weight
            if weight_tensor is None:
                weight_tensor = weights.reshape(-1)
            else:
                weight_tensor = torch.cat((weight_tensor, weights.reshape(-1)))

    weight_tensor = weight_tensor.to(device=device)
    # Train the generator and discriminator using the GAL technique
    for epoch in range(epochs):
        # Generate a batch of synthetic data
        synthetic_data = generator(torch.randn(100, backbone[0].in_channels))

        # Train the discriminator on a combination of real and synthetic data
        discriminator_loss = adversarial_loss(discriminator(weight_tensor), torch.ones_like(discriminator(weight_tensor))) + \
            adversarial_loss(discriminator(synthetic_data.detach()), torch.zeros_like(discriminator(synthetic_data.detach())))
        discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        discriminator_optimizer.step()

        # Train the generator to generate synthetic data that can fool the discriminator
        generator_loss = adversarial_loss(discriminator(generator(torch.randn(100, backbone[0].in_channels))), torch.ones(100, 1))
        generator_optimizer.zero_grad()
        generator_loss.backward()
        generator_optimizer.step()

    # Extract the importance scores from the generator
    importance_scores = generator(torch.randn(1000, backbone[0].in_channels)).mean(dim=0)

    # Prune the backbone based on the importance scores
    num_pruned_channels = int(pruning_ratio * backbone[0].in_channels)
    sorted_indices = torch.argsort(importance_scores)
    pruned_indices = sorted_indices[:num_pruned_channels]
    backbone = nn.Sequential(*[module for i, module in enumerate(backbone) if i not in pruned_indices])

    return backbone
