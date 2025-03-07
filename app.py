import numpy as np
import torch
import torch.nn as nn
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
import pandas as pd

app = Flask(__name__)


# Define the neural network (same as before)
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


# Load the trained model with weights_only=False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BreathModel().to(device)
model.load_state_dict(torch.load('breath_model.pth', map_location=device, weights_only=False))  # Fix applied here
model.eval()

# Load scaler (fit on original data for consistency)
df = pd.read_csv('breatheasy_fixed_data.csv')  # Load your 600k-row data
scaler = StandardScaler()
scaler.fit(df[['Pollen_Level']])  # Fit scaler on Pollen_Level


# Home route - Render the webpage
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    probability = None

    if request.method == 'POST':
        # Collect form data
        features = [
            'Tiredness', 'Dry-Cough', 'Difficulty-in-Breathing', 'Nasal-Congestion', 'Runny-Nose',
            'Age_0-9', 'Age_10-19', 'Age_20-24', 'Age_25-59', 'Age_60+',
            'Gender_Female', 'Gender_Male', 'Pollen_Level', 'Day'
        ]
        user_input = []
        for feature in features:
            if feature == 'Pollen_Level':
                val = float(request.form[feature])
            else:
                val = int(request.form[feature])
            user_input.append(val)

        # Prepare input for model
        user_input = np.array(user_input, dtype=np.float32)
        user_input[-2] = scaler.transform([[user_input[-2]]])[0, 0]  # Scale Pollen_Level
        input_tensor = torch.FloatTensor(user_input).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            pred_output = model(input_tensor)
            pred_label = (pred_output > 0.5).float().item()
            pred_confidence = pred_output.item()

        prediction = int(pred_label)
        probability = pred_confidence

    return render_template('index.html', prediction=prediction, probability=probability)


if __name__ == '__main__':
    app.run(debug=True)