import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Image Transformation
# ============================================================

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])


# ============================================================
# 2. Load Training Dataset
# ============================================================

train_dataset = datasets.GTSRB(
    root="data",
    split="train",
    download=True,
    transform=transform
)

print("Dataset loaded successfully!")
print("Number of training images:", len(train_dataset))
print("Number of classes:", 43)


# ============================================================
# 3. Load Test Dataset
# ============================================================

test_dataset = datasets.GTSRB(
    root="data",
    split="test",
    download=True,
    transform=transform
)

print("Number of test images:", len(test_dataset))


# ============================================================
# 4. Create DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

print("DataLoaders created successfully!")


# ============================================================
# 5. CNN Model
# ============================================================

class TrafficSignCNN(nn.Module):

    def __init__(self):
        super(TrafficSignCNN, self).__init__()

        self.conv1 = nn.Conv2d(
            3, 32,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            32, 64,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(
            64 * 8 * 8,
            128
        )

        self.fc2 = nn.Linear(
            128,
            43
        )

    def forward(self, x):

        x = self.pool(
            F.relu(self.conv1(x))
        )

        x = self.pool(
            F.relu(self.conv2(x))
        )

        x = x.view(
            -1,
            64 * 8 * 8
        )

        x = F.relu(
            self.fc1(x)
        )

        x = self.fc2(x)

        return x


# ============================================================
# 6. Training
# ============================================================

if __name__ == "__main__":

    # Create model
    model = TrafficSignCNN()

    print("CNN model created successfully!")
    print(model)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    print("Loss function and optimizer created successfully!")

    # Number of training epochs
    num_epochs = 5

    # --------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------

    for epoch in range(num_epochs):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            # Clear previous gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Update model weights
            optimizer.step()

            # Calculate loss
            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(
                outputs.data,
                1
            )

            total += labels.size(0)

            correct += (
                (predicted == labels).sum().item()
            )

        accuracy = 100 * correct / total

        average_loss = (
            running_loss / len(train_loader)
        )

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Loss: {average_loss:.4f} "
            f"Accuracy: {accuracy:.2f}%"
        )


    # ========================================================
    # 7. Test the Model
    # ========================================================

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            outputs = model(images)

            _, predicted = torch.max(
                outputs.data,
                1
            )

            total += labels.size(0)

            correct += (
                (predicted == labels).sum().item()
            )

    test_accuracy = 100 * correct / total

    print(
        f"Test Accuracy: {test_accuracy:.2f}%"
    )


    # ========================================================
    # 8. Save Trained Model
    # ========================================================

    torch.save(
        model.state_dict(),
        "models/traffic_sign_cnn.pth"
    )

    print("Model saved successfully!")