from flask import (

    Flask,

    render_template,

    request,

    redirect,

    url_for,

    session,

    flash,

    send_from_directory

)



import sqlite3

import os

import cv2

import numpy as np

import tensorflow as tf

import pandas as pd

from dotenv import load_dotenv

import smtplib

from email.message import EmailMessage

from datetime import datetime

try:

    from zoneinfo import ZoneInfo

except ImportError:

    ZoneInfo = None



# Optional EMAIL + scheduler support for Daily Nutrition Alerts.

# Install with: pip install twilio apscheduler



try:

    from apscheduler.schedulers.background import BackgroundScheduler

except ImportError:

    BackgroundScheduler = None



from werkzeug.security import (

    generate_password_hash,

    check_password_hash

)



from werkzeug.utils import secure_filename





# =========================================================

# 1. FLASK APP

# =========================================================



app = Flask(__name__)



load_dotenv()



app.secret_key = os.getenv("SECRET_KEY", "food_analysis_secret_key")





# =========================================================

# 2. BASE PATH

# =========================================================



BASE_DIR = os.path.dirname(os.path.abspath(__file__))





# =========================================================

# 3. FILE PATHS

# =========================================================



MODEL_PATH = os.path.join(

    BASE_DIR,

    "model",

    "food_model.keras"

)



CLASS_NAMES_PATH = os.path.join(

    BASE_DIR,

    "model",

    "class_names.txt"

)



DATABASE_PATH = os.getenv(
    "DATABASE_PATH",
    os.path.join(
        BASE_DIR,
        "database.db"
    )
)



UPLOAD_FOLDER = os.getenv(
    "UPLOAD_FOLDER",
    os.path.join(
        BASE_DIR,
        "uploads"
    )
)



NUTRITION_PATH = os.path.join(

    BASE_DIR,

    "nutrition",

    "nutrition.csv"

)





# =========================================================

# 4. UPLOAD SETTINGS

# =========================================================



ALLOWED_EXTENSIONS = {

    "png",

    "jpg",

    "jpeg"

}



app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER



os.makedirs(

    UPLOAD_FOLDER,

    exist_ok=True

)





# =========================================================

# 5. LOAD NUTRITION DATA

# =========================================================



print("\nLoading nutrition dataset...")



if not os.path.exists(NUTRITION_PATH):



    print(

        "ERROR: nutrition CSV was not found!"

    )



    print(

        "Expected location:",

        NUTRITION_PATH

    )



    raise FileNotFoundError(

        "nutrition.csv not found in nutrition folder."

    )





nutrition_data = pd.read_csv(

    NUTRITION_PATH

)



print(

    "Nutrition dataset loaded successfully!"

)



print(

    "Number of nutrition records:",

    len(nutrition_data)

)





# =========================================================

# 6. LOAD TRAINED MODEL

# =========================================================



print("\nLoading trained food model...")



if not os.path.exists(MODEL_PATH):



    print(

        "ERROR: food_model.keras was not found!"

    )



    print(

        "Expected location:",

        MODEL_PATH

    )



    raise FileNotFoundError(

        "food_model.keras not found in model folder."

    )





model = tf.keras.models.load_model(

    MODEL_PATH

)



print(

    "Food model loaded successfully!"

)





# =========================================================

# 7. LOAD FOOD CLASS NAMES

# =========================================================



if not os.path.exists(CLASS_NAMES_PATH):



    print(

        "ERROR: class_names.txt was not found!"

    )



    print(

        "Expected location:",

        CLASS_NAMES_PATH

    )



    raise FileNotFoundError(

        "class_names.txt not found in model folder."

    )





with open(

    CLASS_NAMES_PATH,

    "r",

    encoding="utf-8"

) as f:



    class_names = [

        line.strip()

        for line in f.readlines()

        if line.strip()

    ]





print("\nFood classes loaded:")

print(class_names)



print(

    "Number of classes:",

    len(class_names)

)





# =========================================================

# 8. DATABASE CONNECTION

# =========================================================



def get_db_connection():



    conn = sqlite3.connect(

        DATABASE_PATH

    )



    conn.row_factory = sqlite3.Row



    return conn





# =========================================================

# 9. CREATE DATABASE

# =========================================================



def create_database():



    conn = get_db_connection()



    # Users table

    conn.execute("""

        CREATE TABLE IF NOT EXISTS users (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            name TEXT NOT NULL,

            email TEXT UNIQUE NOT NULL,

            password TEXT NOT NULL

        )

    """)



    # Recently analyzed foods table

    conn.execute("""

        CREATE TABLE IF NOT EXISTS analysis_history (



            id INTEGER PRIMARY KEY AUTOINCREMENT,



            user_id INTEGER NOT NULL,



            food_name TEXT NOT NULL,



            confidence REAL,



            calories TEXT,



            protein TEXT,



            carbohydrates TEXT,



            fat TEXT,



            fiber TEXT,



            analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,



            FOREIGN KEY (user_id)

            REFERENCES users(id)



        )

    """)

        # Nutrition profile table

    conn.execute("""

        CREATE TABLE IF NOT EXISTS nutrition_profiles (



            id INTEGER PRIMARY KEY AUTOINCREMENT,



            user_id INTEGER UNIQUE NOT NULL,



            age INTEGER NOT NULL,



            weight REAL NOT NULL,



            height REAL NOT NULL,



            sex TEXT NOT NULL,



            daily_steps INTEGER NOT NULL DEFAULT 0,



            activity_level INTEGER NOT NULL DEFAULT 1,

            email TEXT,



            goal TEXT NOT NULL,



            daily_calories REAL,



            protein REAL,



            carbohydrates REAL,



            fat REAL,



            fiber REAL,



            FOREIGN KEY (user_id)

            REFERENCES users(id)



        )

    """)



    # ---------------------------------------------------------

    # Backward-compatible columns for an existing database.db

    # ---------------------------------------------------------

    columns = [row[1] for row in conn.execute("PRAGMA table_info(nutrition_profiles)").fetchall()]



    if "daily_steps" not in columns:

        conn.execute("ALTER TABLE nutrition_profiles ADD COLUMN daily_steps INTEGER NOT NULL DEFAULT 0")



    if "activity_level" not in columns:
        conn.execute("ALTER TABLE nutrition_profiles ADD COLUMN activity_level INTEGER NOT NULL DEFAULT 1")

    if "email" not in columns:

        conn.execute("ALTER TABLE nutrition_profiles ADD COLUMN email TEXT")



    conn.commit()



    conn.close()



# Initialize database tables on startup
try:
    create_database()
except Exception as _e:
    print("Database init info:", _e)





# =========================================================

# 10. DAILY NUTRITION ALERT HELPERS

# =========================================================



def get_number(value):

    """Convert values such as '120 kcal' or '12.5 g' into a float."""

    try:

        value = str(value).strip().lower()

        value = value.replace("kcal", "").replace(" g", "").replace("g", "")

        return float(value.strip())

    except (ValueError, TypeError):

        return 0.0





def get_today_totals(user_id):

    """Return today's consumed nutrients for one user."""

    conn = get_db_connection()

    rows = conn.execute("""

        SELECT calories, protein, carbohydrates, fat, fiber

        FROM analysis_history

        WHERE user_id = ?

        AND DATE(analyzed_at, 'localtime') = DATE('now', 'localtime')

    """, (user_id,)).fetchall()

    conn.close()



    return {

        "calories": round(sum(get_number(r["calories"]) for r in rows), 1),

        "protein": round(sum(get_number(r["protein"]) for r in rows), 1),

        "carbohydrates": round(sum(get_number(r["carbohydrates"]) for r in rows), 1),

        "fat": round(sum(get_number(r["fat"]) for r in rows), 1),

        "fiber": round(sum(get_number(r["fiber"]) for r in rows), 1),

    }





def send_email(recipient, subject, body):

    """

    Send an email using SMTP settings supplied through environment variables.



    Required:

      FOODNUTRI_SMTP_HOST

      FOODNUTRI_SMTP_PORT

      FOODNUTRI_SMTP_USERNAME

      FOODNUTRI_SMTP_PASSWORD

      FOODNUTRI_FROM_EMAIL



    Example for Gmail:

      host = smtp.gmail.com

      port = 587

      username = your Gmail address

      password = a Gmail App Password

      from = your Gmail address

    """

    host = os.getenv("FOODNUTRI_SMTP_HOST")

    port = int(os.getenv("FOODNUTRI_SMTP_PORT", "587"))

    username = os.getenv("FOODNUTRI_SMTP_USERNAME")

    password = os.getenv("FOODNUTRI_SMTP_PASSWORD")

    from_email = os.getenv("FOODNUTRI_FROM_EMAIL") or username



    if not all([host, username, password, from_email, recipient]):

        return False, "Email settings are not configured."



    try:

        msg = EmailMessage()

        msg["Subject"] = subject

        msg["From"] = from_email

        msg["To"] = recipient

        msg.set_content(body)



        with smtplib.SMTP(host, port, timeout=20) as server:

            server.starttls()

            server.login(username, password)

            server.send_message(msg)



        return True, "Email sent successfully."

    except Exception as exc:

        print("Email error:", exc)

        return False, "Food saved, but the email could not be sent."



def get_profile_with_id(user_id):

    conn = get_db_connection()

    profile = conn.execute("""

        SELECT np.*, u.email

        FROM nutrition_profiles np

        JOIN users u ON u.id = np.user_id

        WHERE np.user_id = ?

    """, (user_id,)).fetchone()

    conn.close()

    return profile





def send_morning_alerts():

    """Send each user's daily target message."""

    conn = get_db_connection()

    profiles = conn.execute("""

        SELECT np.user_id, u.email, np.daily_calories, np.protein

        FROM nutrition_profiles np

        JOIN users u ON u.id = np.user_id

        WHERE u.email IS NOT NULL AND TRIM(u.email) != ''

    """).fetchall()

    conn.close()



    for profile in profiles:

        message = (

            f"FoodNutri: Your daily target is {profile['daily_calories']:.0f} kcal "

            f"and {profile['protein']:.1f}g protein. Have a healthy day!"

        )

        send_email(

            profile["email"],

            "FoodNutri - Daily Nutrition Target",

            message

        )





def send_evening_alerts():

    """Send today's nutrition summary to each configured user."""

    conn = get_db_connection()

    profiles = conn.execute("""

        SELECT np.user_id, u.email, np.daily_calories, np.protein

        FROM nutrition_profiles np

        JOIN users u ON u.id = np.user_id

        WHERE u.email IS NOT NULL AND TRIM(u.email) != ''

    """).fetchall()

    conn.close()



    for profile in profiles:

        totals = get_today_totals(profile["user_id"])

        message = (

            f"FoodNutri: Today's nutrition summary: "

            f"{totals['calories']:.0f}/{profile['daily_calories']:.0f} kcal consumed. "

            f"Protein: {totals['protein']:.1f}/{profile['protein']:.1f}g."

        )

        send_email(

            profile["email"],

            "FoodNutri - Today's Nutrition Summary",

            message

        )





def start_alert_scheduler():

    """Start daily 8 AM and 9 PM alerts when APScheduler is installed."""

    if BackgroundScheduler is None:

        print("APScheduler not installed; automatic EMAIL alerts are disabled.")

        return None



    scheduler_kwargs = {}

    if ZoneInfo is not None:

        scheduler_kwargs["timezone"] = ZoneInfo("Asia/Kolkata")

    scheduler = BackgroundScheduler(**scheduler_kwargs)

    scheduler.add_job(send_morning_alerts, "cron", hour=9, minute=45, id="morning_nutrition_alert", replace_existing=True)

    scheduler.add_job(send_evening_alerts, "cron", hour=21, minute=0, id="evening_nutrition_alert", replace_existing=True)

    scheduler.start()

    print("Daily Nutrition Alert scheduler started: 09:45 and 21:00.")

    return scheduler





# =========================================================

# 11. CHECK FILE EXTENSION

# =========================================================



def allowed_file(filename):



    return (

        "." in filename

        and

        filename.rsplit(

            ".",

            1

        )[1].lower()

        in ALLOWED_EXTENSIONS

    )





# =========================================================

# 11. LOGIN

# =========================================================



@app.route(

    "/",

    methods=["GET", "POST"]

)

def login():



    if request.method == "POST":



        email = request.form.get(

            "email",

            ""

        ).strip()



        password = request.form.get(

            "password",

            ""

        )



        # Check empty fields



        if not email or not password:



            flash(

                "Please enter email and password."

            )



            return redirect(

                url_for("login")

            )





        conn = get_db_connection()



        user = conn.execute(

            """

            SELECT *

            FROM users

            WHERE email = ?

            """,

            (email,)

        ).fetchone()



        conn.close()





        # Check login



        if user and check_password_hash(

            user["password"],

            password

        ):



            session["user_id"] = user["id"]



            session["user_name"] = user["name"]



            return redirect(

                url_for("home")

            )





        flash(

            "Invalid email or password."

        )





    return render_template(

        "login.html"

    )





# =========================================================

# 12. REGISTER

# =========================================================



@app.route(

    "/register",

    methods=["GET", "POST"]

)

def register():



    if request.method == "POST":



        name = request.form.get(

            "name",

            ""

        ).strip()



        email = request.form.get(

            "email",

            ""

        ).strip()



        password = request.form.get(

            "password",

            ""

        )



        confirm_password = request.form.get(

            "confirm_password",

            ""

        )



        # Check empty fields



        if (

            not name

            or not email

            or not password

            or not confirm_password

        ):



            flash(

                "Please fill all fields."

            )



            return redirect(

                url_for("register")

            )





        # Check passwords



        if password != confirm_password:



            flash(

                "Passwords do not match."

            )



            return redirect(

                url_for("register")

            )





        conn = get_db_connection()





        # Check existing user



        existing_user = conn.execute(

            """

            SELECT *

            FROM users

            WHERE email = ?

            """,

            (email,)

        ).fetchone()





        if existing_user:



            conn.close()



            flash(

                "Email already registered."

            )



            return redirect(

                url_for("register")

            )





        # Hash password



        hashed_password = generate_password_hash(

            password

        )





        # Insert user



        conn.execute(

            """

            INSERT INTO users

            (name, email, password)



            VALUES (?, ?, ?)

            """,

            (

                name,

                email,

                hashed_password

            )

        )



        conn.commit()



        conn.close()





        flash(

            "Account created successfully! Please login."

        )



        return redirect(

            url_for("login")

        )





    return render_template(

        "register.html"

    )





# =========================================================

# 13. HOME PAGE

# =========================================================

@app.route("/home")

def home():



    if "user_id" not in session:

        return redirect(url_for("login"))



    conn = get_db_connection()



    # -----------------------------------------------------

    # Recently analyzed foods

    # -----------------------------------------------------

    recent_foods = conn.execute("""

        SELECT *

        FROM analysis_history

        WHERE user_id = ?

        ORDER BY analyzed_at DESC

        LIMIT 5

    """, (session["user_id"],)).fetchall()



    # -----------------------------------------------------

    # Nutrition profile

    # -----------------------------------------------------

    profile = conn.execute("""

        SELECT *

        FROM nutrition_profiles

        WHERE user_id = ?

    """, (session["user_id"],)).fetchone()



    # -----------------------------------------------------

    # Today's analyzed foods

    # -----------------------------------------------------

    today_foods = conn.execute("""

        SELECT calories, protein, carbohydrates, fat, fiber

        FROM analysis_history

        WHERE user_id = ?

        AND DATE(analyzed_at, 'localtime') = DATE('now', 'localtime')

    """, (session["user_id"],)).fetchall()



    conn.close()



    # Convert values like "40", "1.10 g", or "40 kcal" to numbers

    def number(value):

        try:

            value = str(value).strip()

            value = value.replace("kcal", "")

            value = value.replace(" g", "")

            value = value.replace("g", "")

            return float(value.strip())

        except (ValueError, TypeError):

            return 0.0



    # -----------------------------------------------------

    # Today's nutrition totals

    # -----------------------------------------------------

    today_totals = {

        "calories": round(

            sum(number(food["calories"]) for food in today_foods), 1

        ),

        "protein": round(

            sum(number(food["protein"]) for food in today_foods), 1

        ),

        "carbohydrates": round(

            sum(number(food["carbohydrates"]) for food in today_foods), 1

        ),

        "fat": round(

            sum(number(food["fat"]) for food in today_foods), 1

        ),

        "fiber": round(

            sum(number(food["fiber"]) for food in today_foods), 1

        )

    }



    # -----------------------------------------------------

    # Nutrition recommendations

    # -----------------------------------------------------

    recommendations = None

    selected_goal = None



    if profile:



        age = profile["age"]

        weight = profile["weight"]

        height = profile["height"]

        sex = profile["sex"]

        daily_steps = int(profile["daily_steps"] or 0)



        # Normalize the saved goal so the matching card is always highlighted.

        selected_goal = str(profile["goal"]).strip().lower().replace("-", "_")

        selected_goal = " ".join(selected_goal.split())



        if selected_goal in ("balance", "weight balance", "weight_balance"):

            selected_goal = "weight_balance"

        elif selected_goal in ("loss", "weight loss", "weight_loss"):

            selected_goal = "weight_loss"

        elif selected_goal in ("gain", "weight gain", "weight_gain"):

            selected_goal = "weight_gain"



        # BMR

        if sex == "male":

            bmr = (

                10 * weight

                + 6.25 * height

                - 5 * age

                + 5

            )

        else:

            bmr = (

                10 * weight

                + 6.25 * height

                - 5 * age

                - 161

            )



        # Estimate activity from daily steps.

        if daily_steps < 5000:

            activity_factor = 1.2

        elif daily_steps < 7500:

            activity_factor = 1.375

        elif daily_steps < 10000:

            activity_factor = 1.55

        elif daily_steps < 12500:

            activity_factor = 1.725

        else:

            activity_factor = 1.9



        # Estimated maintenance calories

        maintenance_calories = bmr * activity_factor



        # Protein recommendation

        protein = weight * 1.2



        def calculate_nutrients(calories):



            fat = (calories * 0.25) / 9



            carbohydrates = (

                calories

                - (protein * 4)

                - (fat * 9)

            ) / 4



            return {

                "calories": round(calories),

                "protein": round(protein, 1),

                "carbohydrates": round(carbohydrates, 1),

                "fat": round(fat, 1),

                "fiber": 30

            }



        # Three recommendation plans

        recommendations = {

            "weight_loss": calculate_nutrients(

                maintenance_calories - 400

            ),

            "weight_balance": calculate_nutrients(

                maintenance_calories

            ),

            "weight_gain": calculate_nutrients(

                maintenance_calories + 300

            )

        }



    return render_template(

        "home.html",

        name=session["user_name"],

        recent_foods=recent_foods,

        profile=profile,

        selected_goal=selected_goal,

        today_totals=today_totals,

        recommendations=recommendations

    )





# =========================================================

# NUTRITION PROFILE

# =========================================================



@app.route("/nutrition-profile", methods=["GET", "POST"])

def nutrition_profile():



    if "user_id" not in session:

        return redirect(url_for("login"))



    # Get user's email from database

    user_conn = get_db_connection()



    user_row = user_conn.execute(

        "SELECT email FROM users WHERE id = ?",

        (session["user_id"],)

    ).fetchone()



    user_conn.close()



    email = user_row["email"] if user_row and user_row["email"] else None



    if request.method == "POST":



        # Get user input

        age = int(request.form.get("age"))

        weight = float(request.form.get("weight"))

        height = float(request.form.get("height"))



        sex = request.form.get("sex")

        daily_steps = int(request.form.get("daily_steps", 0) or 0)

        goal = request.form.get("goal")





        # ================================================

        # Calculate BMR

        # ================================================



        if sex == "male":



            bmr = (

                10 * weight

                + 6.25 * height

                - 5 * age

                + 5

            )



        else:



            bmr = (

                10 * weight

                + 6.25 * height

                - 5 * age

                - 161

            )





        # ================================================

        # Activity multiplier from daily steps

        # ================================================



        if daily_steps < 5000:

            activity_factor = 1.2

        elif daily_steps < 7500:

            activity_factor = 1.375

        elif daily_steps < 10000:

            activity_factor = 1.55

        elif daily_steps < 12500:

            activity_factor = 1.725

        else:

            activity_factor = 1.9





        # ================================================

        # Estimated maintenance calories

        # ================================================



        maintenance_calories = (

            bmr * activity_factor

        )





        # ================================================

        # Goal adjustment

        # ================================================



        if goal == "weight_loss":



            daily_calories = maintenance_calories - 400



        elif goal == "weight_gain":



            daily_calories = maintenance_calories + 300



        else:



            daily_calories = maintenance_calories





        # ================================================

        # Nutrient recommendations

        # ================================================



        protein = weight * 1.2



        fat = (

            daily_calories * 0.25

        ) / 9



        carbohydrates = (

            daily_calories

            - (protein * 4)

            - (fat * 9)

        ) / 4



        fiber = 30





        # Round values

        daily_calories = round(

            daily_calories

        )



        protein = round(

            protein,

            1

        )



        carbohydrates = round(

            carbohydrates,

            1

        )



        fat = round(

            fat,

            1

        )



        fiber = round(

            fiber,

            1

        )





        # ================================================

        # Save profile

        # ================================================



        conn = get_db_connection()



        existing_profile = conn.execute("""

            SELECT id

            FROM nutrition_profiles

            WHERE user_id = ?

        """, (

            session["user_id"],

        )).fetchone()





        if existing_profile:



            conn.execute("""

                UPDATE nutrition_profiles



                SET

                    age = ?,

                    weight = ?,

                    height = ?,

                    sex = ?,

                    daily_steps = ?,

                    activity_level = ?,

                    email = ?,

                    goal = ?,

                    daily_calories = ?,

                    protein = ?,

                    carbohydrates = ?,

                    fat = ?,

                    fiber = ?



                WHERE user_id = ?

            """, (

                age,

                weight,

                height,

                sex,

                daily_steps,

                1,
                email,

                goal,

                daily_calories,

                protein,

                carbohydrates,

                fat,

                fiber,

                session["user_id"]

            ))



        else:



            conn.execute("""

                INSERT INTO nutrition_profiles
                (
                    user_id,
                    age,
                    weight,
                    height,
                    sex,
                    daily_steps,
                    activity_level,
                    email,
                    goal,
                    daily_calories,
                    protein,
                    carbohydrates,
                    fat,
                    fiber
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session["user_id"],
                age,
                weight,
                height,
                sex,
                daily_steps,
                1,
                email,
                goal,
                daily_calories,
                protein,
                carbohydrates,
                fat,
                fiber
            ))


        conn.commit()

        conn.close()





        return redirect(

            url_for("home")

        )





    return render_template(

        "nutrition_profile.html"

    )



# =========================================================

# 14. SERVE UPLOADED IMAGES

# =========================================================



@app.route(

    "/uploads/<filename>"

)

def uploaded_file(filename):



    return send_from_directory(

        app.config["UPLOAD_FOLDER"],

        filename

    )





# =========================================================

# 15. FOOD IMAGE ANALYSIS

# =========================================================



@app.route("/save-analysis", methods=["POST"])

def save_analysis():



    if "user_id" not in session:

        return redirect(url_for("login"))



    try:

        weight = float(request.form.get("weight", 0))

        if weight <= 0 or weight > 10000:

            raise ValueError



        food_name = request.form.get("food_name", "Unknown Food").strip()

        confidence = float(request.form.get("confidence", "0"))



        # Nutrition values in nutrition.csv are per 100 g.

        # Recalculate them on the server so the database/EMAIL use the

        # authoritative dataset rather than trusting hidden form fields.

        nutrition_mapping = {

            "bacon": "Pork, baked, cooked, bacon, cured",

            "banana": "Bananas, raw",

            "bread": "Bread, white wheat",

            "broccoli": "Broccoli, raw",

            "butter": "Butter, without salt",

            "carrots": "Carrots, raw",

            "cheese": "Cheese, cheddar",

            "chicken": "Chicken, raw, meat only, broilers or fryers",

            "cucumber": "Cucumber, raw, with peel",

            "eggs": "Egg, fresh, raw, whole",

            "fish": "Fish, raw, tilapia",

            "lettuce": "Lettuce, raw, green leaf",

            "milk": "Milk, fluid, low sodium",

            "onions": "Onions, raw",

            "peppers": "Peppers, raw, green, sweet",

            "potatoes": "Potatoes, raw, flesh and skin",

            "sausages": "Beef sausage, cooked, fresh",

            "spinach": "Spinach, raw",

            "tomato": "Tomatoes, year round average, raw, ripe, red",

            "yogurt": "Yogurt, lowfat, plain, Greek"

        }



        def csv_number(value):

            if pd.isna(value):

                raise ValueError

            value = str(value).strip().lower()

            value = value.replace("kcal", "").replace(" g", "").replace("g", "")

            return float(value.strip())



        nutrition_name = nutrition_mapping.get(food_name.lower())

        if not nutrition_name:

            raise ValueError("Nutrition mapping not found for detected food.")



        match = nutrition_data[

            nutrition_data["name"].astype(str).str.lower() == nutrition_name.lower()

        ]



        if match.empty:

            raise ValueError("Nutrition data not found for detected food.")



        row = match.iloc[0]



        per100 = {

            "calories": csv_number(row["calories"]),

            "protein": csv_number(row["protein"]),

            "carbohydrates": csv_number(row["carbohydrate"]),

            "fat": csv_number(row["fat"]),

            "fiber": csv_number(row["fiber"])

        }



        factor = weight / 100.0

        actual = {

            key: round(value * factor, 2)

            for key, value in per100.items()

        }



        conn = get_db_connection()



        conn.execute("""

            INSERT INTO analysis_history

            (

                user_id,

                food_name,

                confidence,

                calories,

                protein,

                carbohydrates,

                fat,

                fiber

            )

            VALUES (?, ?, ?, ?, ?, ?, ?, ?)

        """, (

            session["user_id"],

            food_name,

            round(confidence, 2),

            f"{actual['calories']:.2f}",

            f"{actual['protein']:.2f}",

            f"{actual['carbohydrates']:.2f}",

            f"{actual['fat']:.2f}",

            f"{actual['fiber']:.2f}"

        ))



        conn.commit()

        conn.close()



        # Send the requested real-time food-added email.

        profile = get_profile_with_id(session["user_id"])

        email_status = None



        if profile and profile["email"]:

            totals = get_today_totals(session["user_id"])

            daily_calories = float(profile["daily_calories"] or 0)

            remaining = max(0, daily_calories - totals["calories"])



            email_message = (

                "FoodNutri - Food Added\n"

                f"Food: {food_name}\n"

                f"Quantity: {weight:g} g\n\n"

                f"Calories: {actual['calories']:.2f} kcal\n"

                f"Protein: {actual['protein']:.2f} g\n"

                f"Carbohydrates: {actual['carbohydrates']:.2f} g\n"

                f"Fat: {actual['fat']:.2f} g\n"

                f"Fiber: {actual['fiber']:.2f} g\n\n"

                "This food has been added to Today's Nutrition.\n"

                f"Today's total: {totals['calories']:.2f} kcal\n"

                f"Remaining target: {remaining:.2f} kcal"

            )



            ok, email_status = send_email(

                profile["email"],

                "FoodNutri - Food Added",

                email_message

            )



        if email_status:

            flash(f"{food_name} ({weight:g} g) added. {email_status}")

        else:

            flash(f"{food_name} ({weight:g} g) added to Today's Nutrition.")



        return redirect(url_for("home"))



    except (ValueError, TypeError, KeyError) as exc:

        print("Save analysis error:", exc)

        flash("Unable to save this food. Please enter a valid weight and try again.")

        return redirect(url_for("home"))





@app.route(

    "/analyze",

    methods=["POST"]

)

def analyze():



    # -----------------------------------------------------

    # Check login

    # -----------------------------------------------------



    if "user_id" not in session:



        return redirect(

            url_for("login")

        )





    # -----------------------------------------------------

    # Check uploaded file

    # -----------------------------------------------------



    if "food_image" not in request.files:



        flash(

            "Please select a food image."

        )



        return redirect(

            url_for("home")

        )





    file = request.files["food_image"]





    # -----------------------------------------------------

    # Check filename

    # -----------------------------------------------------



    if file.filename == "":



        flash(

            "Please select a food image."

        )



        return redirect(

            url_for("home")

        )





    # -----------------------------------------------------

    # Check extension

    # -----------------------------------------------------



    if not allowed_file(

        file.filename

    ):



        flash(

            "Only JPG, JPEG and PNG images are allowed."

        )



        return redirect(

            url_for("home")

        )





    # -----------------------------------------------------

    # Secure filename

    # -----------------------------------------------------



    filename = secure_filename(

        file.filename

    )





    # -----------------------------------------------------

    # Save image

    # -----------------------------------------------------



    filepath = os.path.join(

        app.config["UPLOAD_FOLDER"],

        filename

    )



    file.save(filepath)





    # =====================================================

    # 16. OPENCV PROCESSING

    # =====================================================



    image = cv2.imread(

        filepath

    )





    # Check image



    if image is None:



        flash(

            "Unable to read the uploaded image."

        )



        return redirect(

            url_for("home")

        )





    # -----------------------------------------------------

    # Get original dimensions

    # -----------------------------------------------------



    height, width = image.shape[:2]





    # -----------------------------------------------------

    # Resize image

    # -----------------------------------------------------



    resized_image = cv2.resize(

        image,

        (224, 224)

    )





    # -----------------------------------------------------

    # Convert BGR to RGB

    # -----------------------------------------------------



    rgb_image = cv2.cvtColor(

        resized_image,

        cv2.COLOR_BGR2RGB

    )





    # =====================================================

    # 17. PREPARE IMAGE FOR CNN

    # =====================================================



    input_image = np.array(

        rgb_image,

        dtype=np.float32

    )





    # Add batch dimension



    input_image = np.expand_dims(

        input_image,

        axis=0

    )





    # =====================================================

    # 18. CNN PREDICTION

    # =====================================================



    print("\nMaking prediction...")





    predictions = model.predict(

        input_image,

        verbose=0

    )





    # -----------------------------------------------------

    # Find highest probability

    # -----------------------------------------------------



    predicted_index = int(

        np.argmax(

            predictions[0]

        )

    )





    # -----------------------------------------------------

    # Get confidence

    # -----------------------------------------------------



    confidence_value = float(

        predictions[0][predicted_index]

    ) * 100





    # -----------------------------------------------------

    # Get food name

    # -----------------------------------------------------



    if predicted_index < len(class_names):



        detected_food = class_names[

            predicted_index

        ]



    else:



        detected_food = "Unknown Food"





    confidence = round(

        confidence_value,

        2

    )





       # =====================================================

    # 19. NUTRITION INFORMATION

    # =====================================================



    # Exact nutrition mapping for the 20 CNN food classes

    nutrition_mapping = {



        "bacon": "Pork, baked, cooked, bacon, cured",



        "banana": "Bananas, raw",



        "bread": "Bread, white wheat",



        "broccoli": "Broccoli, raw",



        "butter": "Butter, without salt",



        "carrots": "Carrots, raw",



        "cheese": "Cheese, cheddar",



        "chicken": "Chicken, raw, meat only, broilers or fryers",



        "cucumber": "Cucumber, raw, with peel",



        "eggs": "Egg, fresh, raw, whole",



        "fish": "Fish, raw, tilapia",



        "lettuce": "Lettuce, raw, green leaf",



        "milk": "Milk, fluid, low sodium",



        "onions": "Onions, raw",



        "peppers": "Peppers, raw, green, sweet",



        "potatoes": "Potatoes, raw, flesh and skin",



        "sausages": "Beef sausage, cooked, fresh",



        "spinach": "Spinach, raw",



        "tomato": "Tomatoes, year round average, raw, ripe, red",



        "yogurt": "Yogurt, lowfat, plain, Greek"

    }





    # Convert detected food to lowercase

    food_search = detected_food.lower().strip()





    # Default values

    calories = "N/A"

    protein = "N/A"

    carbohydrates = "N/A"

    fat = "N/A"

    fiber = "N/A"





    # Get the exact nutrition dataset name

    nutrition_name = nutrition_mapping.get(

        food_search

    )





    # Find exact nutrition row

    if nutrition_name is not None:



        nutrition_match = nutrition_data[

            nutrition_data["name"].astype(str).str.lower()

            == nutrition_name.lower()

        ]



        if not nutrition_match.empty:



            nutrition_row = nutrition_match.iloc[0]



            calories = nutrition_row["calories"]

            protein = nutrition_row["protein"]

            carbohydrates = nutrition_row["carbohydrate"]

            fat = nutrition_row["fat"]

            fiber = nutrition_row["fiber"]





                # =====================================================



    # =====================================================

    # Remove units already present in CSV

    # =====================================================



    def clean_nutrient(value):



        if pd.isna(value):



            return "N/A"



        value = str(value).strip()



        value = value.replace(" g", "")

        value = value.replace("g", "")



        return value.strip()





    protein = clean_nutrient(protein)



    carbohydrates = clean_nutrient(carbohydrates)



    fat = clean_nutrient(fat)



    fiber = clean_nutrient(fiber)



    calories = clean_nutrient(calories)



    # =====================================================

    # 20. PRINT RESULT

    # =====================================================



    print(

        "Predicted Food:",

        detected_food

    )



    print(

        "Confidence:",

        confidence,

        "%"

    )



    print(

        "Calories:",

        calories

    )



    print(

        "Protein:",

        protein

    )



    print(

        "Carbohydrates:",

        carbohydrates

    )



    print(

        "Fat:",

        fat

    )



    print(

        "Fiber:",

        fiber

    )





    # =====================================================

    # 21. SHOW RESULT PAGE

    # =====================================================



    return render_template(

        "result.html",



        image_file=filename,



        food_name=detected_food,



        confidence=confidence,



        width=width,



        height=height,



        calories=calories,



        protein=protein,



        carbohydrates=carbohydrates,



        fat=fat,



        fiber=fiber

    )





# =========================================================

# 22. DAILY NUTRITION ALERT TEST ROUTES

# =========================================================



@app.route("/send-test-morning-alert")

def send_test_morning_alert():

    if "user_id" not in session:

        return redirect(url_for("login"))



    profile = get_profile_with_id(session["user_id"])

    if not profile or not profile["email"]:

        flash("Please save a email address in your Nutrition Profile first.")

        return redirect(url_for("home"))



    message = (

        f"FoodNutri: Your daily target is {profile['daily_calories']:.0f} kcal "

        f"and {profile['protein']:.1f}g protein. Have a healthy day!"

    )

    ok, status = send_email(profile["email"],"Foodnutri - Test Nutrition Alert", message)

    flash(status)

    return redirect(url_for("home"))





@app.route("/send-test-summary-alert")

def send_test_summary_alert():

    if "user_id" not in session:

        return redirect(url_for("login"))



    profile = get_profile_with_id(session["user_id"])

    if not profile or not profile["email"]:

        flash("Please save a email address in your Nutrition Profile first.")

        return redirect(url_for("home"))



    totals = get_today_totals(session["user_id"])

    message = (

        f"FoodNutri: Today's nutrition summary: "

        f"{totals['calories']:.0f}/{profile['daily_calories']:.0f} kcal consumed. "

        f"Protein: {totals['protein']:.1f}/{profile['protein']:.1f}g."

    )

    ok, status = send_email(profile["email"], message)

    flash(status)

    return redirect(url_for("home"))





# =========================================================

# 23. LOGOUT

# =========================================================



@app.route("/logout")

def logout():



    session.clear()



    return redirect(

        url_for("login")

    )





# =========================================================

# 23. RUN APPLICATION

# =========================================================



if __name__ == "__main__":



    create_database()



    print("\n====================================")

    print("FoodNutri - Real-Time Nutrition Monitoring + EMAIL Alerts")

    print("====================================")



    print(

        "Starting server...\n"

    )



    start_alert_scheduler()



    port = int(os.getenv("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=os.getenv("FLASK_DEBUG", "False").lower() in ("true", "1"),
        use_reloader=False
    )