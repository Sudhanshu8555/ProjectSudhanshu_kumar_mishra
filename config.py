import torch

batch_size = 32
num_epochs = 100
learning_rate = 0.001
weight_decay = 1e-4
patience = 10
resize_x = 224
resize_y = 224
input_channels = 3
num_classes = 10

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Paths
train_dir = "data"    # replace this with your training path
val_dir = "data"      # replace this with your validation path
test_dir = "data"     # replace this woth your test path
