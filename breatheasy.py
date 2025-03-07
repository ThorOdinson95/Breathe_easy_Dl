import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Neural network definition
class BreathModel(nn.Module):
    def __init__(self, feature_count=14):
        super(BreathModel, self).__init__()
        self.layer1 = nn.Linear(feature_count, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.layer4 = nn.Linear(32, 1)
        self.activation = nn.ReLU()
        self.output_activation = nn.Sigmoid()
        self.regularizer = nn.Dropout(0.3)

    def forward(self, data):
        data = self.activation(self.layer1(data))
        data = self.regularizer(data)
        data = self.activation(self.layer2(data))
        data = self.regularizer(data)
        data = self.activation(self.layer3(data))
        data = self.regularizer(data)
        data = self.output_activation(self.layer4(data))
        return data


# Function to collect user-defined synthetic data
def collect_user_data(normalizer):
    print("\nProvide synthetic data for prediction (use 0/1 for binary, float for Pollen_Level):")
    feature_list = [
        'Tiredness', 'Dry-Cough', 'Difficulty-in-Breathing', 'Nasal-Congestion', 'Runny-Nose',
        'Age_0-9', 'Age_10-19', 'Age_20-24', 'Age_25-59', 'Age_60+',
        'Gender_Female', 'Gender_Male', 'Pollen_Level', 'Day'
    ]
    user_input = []
    for feat in feature_list:
        if feat == 'Pollen_Level':
            value = float(input(f"{feat} (e.g., 0-10): "))
        else:
            value = int(input(f"{feat} (enter 0 or 1): "))
            if value not in [0, 1] and feat != 'Pollen_Level':
                raise ValueError(f"{feat} should be either 0 or 1")
        user_input.append(value)

    # Transform into array and normalize Pollen_Level
    user_input = np.array(user_input, dtype=np.float32)
    user_input[-2] = normalizer.transform([[user_input[-2]]])[0, 0]  # Adjust Pollen_Level
    return torch.FloatTensor(user_input).unsqueeze(0)  # Add batch dimension


if __name__ == '__main__':
    # Import the dataset
    print("Fetching data...")
    data_frame = pd.read_csv('breatheasy_fixed_data.csv')  # Use your 600k-row dataset path

    # Separate features and labels
    inputs = data_frame.drop('Attack_Risk', axis=1).values
    labels = data_frame['Attack_Risk'].values

    # Divide into training and testing sets
    train_X, test_X, train_y, test_y = train_test_split(inputs, labels, test_size=0.2, random_state=42)

    # Normalize the continuous feature (Pollen_Level)
    normalizer = StandardScaler()
    train_X[:, -2] = normalizer.fit_transform(train_X[:, -2].reshape(-1, 1)).flatten()
    test_X[:, -2] = normalizer.transform(test_X[:, -2].reshape(-1, 1)).flatten()

    # Prepare tensors for PyTorch
    train_X_tensor = torch.FloatTensor(train_X)
    test_X_tensor = torch.FloatTensor(test_X)
    train_y_tensor = torch.FloatTensor(train_y).unsqueeze(1)
    test_y_tensor = torch.FloatTensor(test_y).unsqueeze(1)

    # Set up data loader for batch processing
    training_set = TensorDataset(train_X_tensor, train_y_tensor)
    batch_loader = DataLoader(training_set, batch_size=1024, shuffle=True, num_workers=4)  # Tune num_workers as needed

    # Configure model, loss function, and optimizer
    computing_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {computing_device}")
    neural_net = BreathModel().to(computing_device)
    loss_function = nn.BCELoss()
    opt = optim.Adam(neural_net.parameters(), lr=0.001)

    # Train the model
    total_epochs = 20
    print("Beginning training...")
    for epoch in range(total_epochs):
        neural_net.train()
        total_loss = 0.0
        for batch_X, batch_y in batch_loader:
            batch_X, batch_y = batch_X.to(computing_device), batch_y.to(computing_device)
            opt.zero_grad()
            predictions = neural_net(batch_X)
            loss = loss_function(predictions, batch_y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch_X.size(0)

        avg_loss = total_loss / len(batch_loader.dataset)
        print(f'Epoch {epoch + 1}/{total_epochs}, Average Loss: {avg_loss:.4f}')

    # Assess model performance
    neural_net.eval()
    print("Assessing performance...")
    with torch.no_grad():
        test_X_tensor = test_X_tensor.to(computing_device)
        test_predictions = neural_net(test_X_tensor)
        test_predictions = (test_predictions > 0.5).float().cpu().numpy()
        test_labels = test_y_tensor.numpy()

    # Display performance metrics
    acc = accuracy_score(test_labels, test_predictions)
    print(f"\nModel Accuracy on Test Set: {acc:.4f}")
    print("\nPerformance Summary:\n", classification_report(test_labels, test_predictions))
    print("\nConfusion Matrix:\n", confusion_matrix(test_labels, test_predictions))

    # Store the trained model
    torch.save(neural_net.state_dict(), 'breath_model.pth')
    print("Trained model stored as 'breath_model.pth'")

    # Predict with user-provided synthetic data
    neural_net.eval()
    with torch.no_grad():
        user_data = collect_user_data(normalizer).to(computing_device)
        pred_output = neural_net(user_data)
        pred_label = (pred_output > 0.5).float().item()
        pred_confidence = pred_output.item()
        print(f"\nPredicted Outcome: Attack_Risk = {int(pred_label)} (Confidence: {pred_confidence:.4f})")