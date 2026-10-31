---
title: FoodNutri AI Food Detection
emoji: 🥗
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# 🥗 FoodNutri – AI-Based Food Detection and Personalized Nutrition Analysis

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask%203.x-green.svg)](https://flask.palletsprojects.com/)
[![ML Engine](https://img.shields.io/badge/ML-TensorFlow%20%7C%20Keras%20MobileNetV2-orange.svg)](https://www.tensorflow.org/)
[![Database](https://img.shields.io/badge/Database-SQLite3-lightgrey.svg)](https://www.sqlite.org/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

FoodNutri is an intelligent, full-stack nutrition tracking web application powered by Deep Learning. It classifies food images using a fine-tuned **MobileNetV2** Convolutional Neural Network and instantly computes macronutrient details (calories, protein, carbohydrates, fat, and fiber) using a comprehensive nutrition dataset of over 8,700 foods.

---

## 📌 Features

- **🤖 AI-Powered Food Recognition**: MobileNetV2 model fine-tuned on 20 food categories (bacon, banana, bread, broccoli, butter, carrots, cheese, chicken, cucumber, eggs, fish, lettuce, milk, onions, peppers, potatoes, sausages, spinach, tomato, yogurt).
- **📊 Real-Time Nutrition Analysis**: Automatically retrieves calories, protein, carbs, fats, and fiber per 100g, with visual macro charts powered by Chart.js.
- **⚖️ Dynamic Portion Calculator**: Adjust food weight in grams to calculate exact consumed nutrients for custom portion sizes.
- **👤 User Authentication & Profiles**: Secure password hashing with Werkzeug; individualized nutrition profiles factoring in age, weight, height, gender, daily steps, and fitness goals.
- **📈 Daily Nutrition Tracking**: Aggregates today's consumed nutrients against your daily personalized targets.
- **📧 Automated Email Notifications**: Background scheduler sends morning target reminders and evening nutrition summary emails via SMTP.

---

## 🛠️ Tech Stack

- **Backend:** Python 3.12, Flask, Gunicorn, Werkzeug
- **Machine Learning & Vision:** TensorFlow, Keras, OpenCV Headless, NumPy
- **Data & Storage:** SQLite3, Pandas
- **Scheduler & Notifications:** APScheduler, Python `smtplib` / `email`
- **Frontend:** Jinja2 Templates, HTML5, CSS3, Chart.js

---

## 📁 Project Structure

```text
FoodNutri/
├── app.py                   # Main Flask web application and API routes
├── database.db              # SQLite database (auto-created on startup)
├── Dockerfile               # Production container definition
├── Procfile                 # Process file for PaaS deployments (Render, Railway)
├── render.yaml              # Render blueprint deployment configuration
├── vercel.json              # Vercel serverless routing configuration
├── requirements.txt         # Production Python dependencies
├── .env.example             # Template for required environment variables
├── .gitignore               # Ignored files, databases, datasets, and secrets
├── model/
│   ├── food_model.keras     # Trained MobileNetV2 Keras model (~11.6 MB)
│   └── class_names.txt      # 20 target food classification classes
├── nutrition/
│   └── nutrition.csv        # Nutritional dataset (8,789 items with macros)
├── static/
│   └── style.css            # Custom CSS styling
├── templates/
│   ├── home.html            # Dashboard and image upload page
│   ├── login.html           # User login page
│   ├── register.html        # New user registration page
│   ├── nutrition_profile.html # Personalized target profile setup
│   └── result.html          # Prediction results, charts & portion calculator
├── uploads/                 # Temporary storage for uploaded food images
├── test_prediction.py       # Standalone test script for ML model inference
└── train_model.py           # Model training & transfer learning pipeline
```

---

## ⚙️ Environment Variables

Create a `.env` file in the project root based on `.env.example`:

| Variable | Description | Default |
| :--- | :--- | :--- |
| `SECRET_KEY` | Flask session encryption key | `food_analysis_secret_key` |
| `FLASK_DEBUG` | Enable/disable Flask debug mode | `False` |
| `PORT` | Web server listening port | `5000` |
| `DATABASE_PATH` | Path to SQLite database file | `database.db` |
| `UPLOAD_FOLDER` | Directory for uploaded food images | `uploads/` |
| `FOODNUTRI_SMTP_HOST` | SMTP server address | `smtp.gmail.com` |
| `FOODNUTRI_SMTP_PORT` | SMTP port | `587` |
| `FOODNUTRI_SMTP_USERNAME` | SMTP username / sender email | _None_ |
| `FOODNUTRI_SMTP_PASSWORD` | SMTP password / App password | _None_ |
| `FOODNUTRI_FROM_EMAIL` | Displayed sender email address | `$FOODNUTRI_SMTP_USERNAME` |

---

## 🚀 Local Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Pavankumar054/FoodNutri.git
   cd FoodNutri
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux / macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your preferred secret key and SMTP settings
   ```

5. **Run the application:**
   ```bash
   python app.py
   ```
   Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your web browser.

---

## ☁️ Deployment Architecture & Hosting Compatibility

### Why Serverless Platforms (e.g. Vercel) have limitations for this project:
1. **Model & Dependency Bundle Size**: Full TensorFlow + Keras exceeds 500 MB uncompressed. Vercel serverless functions enforce a strict **250 MB maximum uncompressed size** limit on total code and packages.
2. **Stateless / Ephemeral Filesystem**: Vercel functions execute in isolated, ephemeral lambda containers. Local SQLite (`database.db`) writes and image uploads in `/tmp` are wiped between cold starts and are not shared across instances.
3. **Background Daemon Restrictions**: Serverless functions freeze execution immediately after returning an HTTP response; long-running background tasks (such as APScheduler email alerts) cannot run continuously.

### Recommended Production Deployment:
For complete functionality including TensorFlow inference, persistent SQLite/PostgreSQL storage, uploaded image serving, and APScheduler daily email alerts, deploy to a container or Python web service:

- **Render (Recommended)**: Free Web Service running Python with Gunicorn (`Procfile` and `render.yaml` included).
- **Railway**: Native container deployment with persistent volume support (`Dockerfile` included).
- **Hugging Face Spaces**: Free CPU/GPU ML hosting with Docker.
- **Vercel**: Can be used as a frontend proxy routing API requests to a Render/Railway backend.

### Deploying to Render in 3 Steps:
1. Fork or push this repository to your GitHub account.
2. Log in to [Render](https://render.com) and click **New > Web Service**.
3. Select `Pavankumar054/FoodNutri`. Render will automatically detect the Python environment and run `gunicorn app:app`. Add your SMTP environment variables in the Render dashboard.

---

## 📄 License

This project is licensed under the MIT License.
