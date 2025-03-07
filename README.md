# Breathe_easy_Dl
Flask app for asthma risk prediction with PyTorch model and dataset.
# Asthma Predictor

Hey there! Welcome to **Asthma Predictor**, a little passion project I built to help figure out asthma attack risks using some cool tech. Imagine a simple web tool where you punch in a few details—like whether you’re feeling tired or dealing with a dry cough—and it tells you if an asthma attack might be around the corner. That’s what this is all about!

I used Python to whip this up, with a PyTorch neural network doing the heavy lifting under the hood and Flask serving up a clean, user-friendly webpage. It’s trained on a hefty dataset, so it’s got a solid foundation to make those predictions.

---

## What’s Inside?

- **`app.py`**: The heart of the web app—runs the Flask server and loads the trained model for predictions.
- **`breatheasy.py`**: The training script I used to build the model from scratch. Want to tweak it? Go for it!
- **`breatheasy_fixed_data.csv`**: The synthetic dataset—600k rows of asthma-related features like symptoms, age, and pollen levels.
- **`breath_model.pth`**: The pre-trained model weights, ready to roll for instant predictions.
- **`templates/index.html`**: The webpage layout—simple dropdowns and a button to get your result.
- **`static/style.css`**: Some custom styling to make the page look sharp.

---

## How It Works

1. **Training**: I fed the dataset into a PyTorch neural network (think layers of 14→128→64→32→1) using `retrain.py`. It learned patterns from symptoms and demographics to predict attack risk.
2. **Web App**: Fire up `app.py`, and it loads `breath_model.pth`. Enter your details on the webpage, hit "Predict," and boom—you get a risk score (0 or 1) with a probability.
3. **Output**: The page tells you if it’s “Low risk” (green) or “High risk” (red), based on the model’s confidence.

---

## Screenshots

Here’s what it looks like in action:

### The Webpage

*Just a clean form—pick your symptoms and details, then hit Predict!*
![image](https://github.com/user-attachments/assets/869f2f39-fe77-41bb-93be-6b83cc732254)

### model evaluation
![image](https://github.com/user-attachments/assets/02d08826-4b64-4ea7-ae81-a28de8ffaad0)



---

## Running It Yourself

1. **Clone the Repo**:
   ```bash
   git clone https://github.com/yourusername/AsthmaPredictor.git
   cd AsthmaPredictor
