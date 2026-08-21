# Churn-Prediction-Using-ANN

https://churn-prediction-using-ann-blush.vercel.app

# ✦ ChurnGuard AI

**Customer Retention Intelligence Platform powered by an Artificial Neural Network (ANN)**

ChurnGuard AI is a single-file Flask web application that serves real-time customer churn risk predictions using a trained Keras Sequential ANN (`ANN.pkl`). It features a multi-theme interface, sensitivity analysis visualizers, and an analytical monitoring dashboard with zero external CDN dependencies.

---

## 🚀 Features

* **Trained ANN Integration:** Directly loads and evaluates customer data via `ANN.pkl` (Dense 8→8→7→8→7→1 architecture).
* **4 Built-in Themes:** Midnight Gold, Emerald Vault, Royal Amethyst, and Arctic Ivory.
* **Sensitivity Analysis:** Perturbs input features by $\pm 1\sigma$ to show which specific factors impact the customer's churn risk the most.
* **Analytical Dashboard:** Live tracking of session predictions, average churn rate, probability distribution histogram, and recent inference history.
* **Adaptive Preprocessing:** Standardizes inputs using standard dataset statistics, and automatically integrates `scaler.pkl` if present in the directory.

---

## 📁 Repository Structure

```text
├── app.py              # Main Flask application (Frontend + Backend)
├── requirements.txt    # Python package dependencies
├── ANN.pkl             # Trained Keras Sequential Neural Network model
├── scaler.pkl          # (Optional) Pre-trained StandardScaler object
└── README.md           # Documentation
🛠️ Local Setup & Installation1. Clone & NavigateBashgit clone <your-repo-url>
cd <your-repo-folder>
2. Set Up Virtual Environment (Python 3.11 recommended)Bashpython -m venv venv
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
3. Install DependenciesBashpip install -r requirements.txt
4. Run the ApplicationBashpython app.py
Open http://127.0.0.1:5000 in your browser.☁️ Deployment Guide (Render)Because TensorFlow requires Python $\le$ 3.12, Render's default Python version must be pinned.Push app.py, requirements.txt, and ANN.pkl to your GitHub repository.In Render Dashboard:Click New + $\rightarrow$ Web Service.Connect your GitHub repository.Configure the service settings:Runtime: Python 3Build Command: pip install -r requirements.txtStart Command: gunicorn app:app --workers 1 --threads 4 --timeout 120Add the Python Environment Variable:Go to Environment $\rightarrow$ Add Environment Variable.Key: PYTHON_VERSIONValue: 3.11.9Click Deploy Web Service (or use Manual Deploy $\rightarrow$ Clear build cache & deploy).🧠 Feature SchemaThe model evaluates a 10-feature vector:FeatureTypeRange / OptionsCredit ScoreNumerical300 – 850GeographyCategoricalFrance (0), Germany (1), Spain (2)GenderCategoricalFemale (0), Male (1)AgeNumerical18 – 92 yearsTenureNumerical0 – 10 yearsAccount BalanceNumerical$0.00 – $250,000.00Number of ProductsDiscrete1 – 4Has Credit CardBinaryYes (1) / No (0)Active MemberBinaryYes (1) / No (0)Estimated SalaryNumerical$0.00 – $250,000.00
