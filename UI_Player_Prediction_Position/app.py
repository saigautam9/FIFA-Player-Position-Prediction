from flask import Flask, render_template, request
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors

# Initialize Spark Session
spark = SparkSession.builder.appName("PlayerPositionDemo").getOrCreate()

# Load the trained model
model_path = "/Users/arunajithesh/Desktop/SJSU/sem-2/model/RandomForestClassifierModel_FIFA_position_test1"
model = RandomForestClassificationModel.load(model_path)

# Initialize Flask app
app = Flask(__name__)

# Home route to display the form
@app.route("/")
def home():
    # Render the page without any prediction result initially
    return render_template("index.html", position=None)

# Prediction route
# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect the inputs from the form
        pace = float(request.form["pace"])
        stamina = float(request.form["stamina"])
        shooting = float(request.form["shooting"])
        passing = float(request.form["passing"])
        finishing = float(request.form["finishing"])
        defending = float(request.form["defending"])
        tackling = float(request.form["tackling"])

        # Calculate additional features if they exist in the model
        # Weighted defense strength with stamina included
        defense_strength = 0.5 * defending + 0.3 * tackling + 0.2 * stamina

         # Adjusted attack-to-defense ratio with weighted components
        attack_to_defense_ratio = (0.5 * shooting + 0.3 * passing + 0.2 * finishing) / (defending + 1)


        # Use these inputs to infer missing features or use default averages
        feature_values = [
            pace, shooting, passing,
            (shooting + passing + finishing) / 3,  # Dribbling (average approximation)
            defending,  # Defending
            passing,  # Crossing (approximation)
            finishing,  # Finishing
            (passing + finishing) / 2,  # Heading accuracy (approximation)
            passing,  # Short passing (approximation)
            70,  # Volleys (default value)
            stamina,  # Shot power
            stamina,  # Stamina
            shooting,  # Long shots
            tackling,  # Interceptions (approximation)
            defending,  # Positioning (approximation)
            60,  # Penalties (default value)
            tackling,  # Marking awareness (approximation)
            tackling,  # Standing tackle (approximation)
            tackling,  # Sliding tackle (approximation)
            defense_strength,  # Add engineered feature
            attack_to_defense_ratio  # Add engineered feature
        ]

        # Create input DataFrame
        input_data = spark.createDataFrame([(Vectors.dense(feature_values),)], ["features"])

        # Debug: Log the feature values
        print(f"Feature values: {feature_values}")

        # Predict using the trained model
        prediction = model.transform(input_data).select("probability", "prediction").collect()

        # Debug: Log probabilities and predicted class
        probabilities = prediction[0][0]  # Probabilities for each class
        predicted_class = prediction[0][1]  # Predicted class
        print(f"Probabilities: {probabilities}, Predicted Class: {predicted_class}")

        # Map prediction to position
        position_map = {0: "Forward", 1: "Midfielder", 2: "Defender"}
        position = position_map.get(predicted_class, "Unknown")

        # Render the template with the prediction result
        return render_template("index.html", position=position)

    except Exception as e:
        # Handle errors gracefully and display them on the page
        return render_template("index.html", position="Error occurred: " + str(e))



if __name__ == "__main__":
    app.run(debug=True)
