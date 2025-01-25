from flask import Flask, request, render_template,render_template,url_for,redirect,flash # type: ignore
from forms import RegistrationForm,LoginForm
import numpy as np # type: ignore
import pickle
import io


# Importing models and scalers
try:
    model = pickle.load(open('model.pkl', 'rb'))
    sc = pickle.load(open('standscaler.pkl', 'rb'))
    ms = pickle.load(open('minmaxscaler.pkl', 'rb'))
    dtr = pickle.load(open('dtr.pkl', 'rb'))
    preprocesser = pickle.load(open('preprocesser.pkl', 'rb'))
except FileNotFoundError as e:
    raise RuntimeError(f"Required file is missing: {e}")

# Creating Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'a3f5e6d7c8b9a0d1e2f3a4b5c6d7e8f9'

# Home route (index.html)
@app.route('/')
def home():
    return render_template("home.html")

# Crop prediction route
@app.route("/predict_crop", methods=['GET','POST'])
def predict_crop():
    try:
        # Get form inputs
        N = float(request.form.get('Nitrogen', 0))
        P = float(request.form.get('Phosphorus', 0))
        K = float(request.form.get('Potassium', 0))
        temp = float(request.form.get('Temperature', 0))
        humidity = float(request.form.get('Humidity', 0))
        ph = float(request.form.get('Ph', 0))
        rainfall = float(request.form.get('Rainfall', 0))

        # Prepare features
        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        # Transform features
        scaled_features = ms.transform(single_pred)
        final_features = sc.transform(scaled_features)
        prediction = model.predict(final_features)

        # Map prediction to crop
        crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 
                     7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 
                     12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 
                     17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas", 20: "Kidneybeans", 
                     21: "Chickpea", 22: "Coffee"}

        crop = crop_dict.get(prediction[0], "Unknown crop")
        result = f"{crop} is the best crop to be cultivated right there."
    except Exception as e:
        result = f"Error: {e}"
    return render_template('index.html', result=result)


# Crop yield prediction route
@app.route("/predict_yield", methods=['GET', 'POST'])
def predict_yield():
    prediction_value = None  # Initialize prediction_value
    
    if request.method == 'POST':
        try:
            # Get form data
            Year = int(request.form['Year'])  # Ensure it's an integer
            average_rain_fall_mm_per_year = float(request.form['average_rain_fall_mm_per_year'])  # Ensure it's a float
            pesticides_tonnes = float(request.form['pesticides_tonnes'])  # Ensure it's a float
            avg_temp = float(request.form['avg_temp'])  # Ensure it's a float
            Area = request.form['Area']  # Assume Area is a string, no conversion needed
            Item = request.form['Item']  # Assume Item is a string, no conversion needed

            # Prepare the features (ensure that all inputs are correctly formatted)
            features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)
            transformed_features = preprocesser.transform(features)  # Assuming preprocesser is correctly defined

            # Make prediction
            prediction_value = dtr.predict(transformed_features).reshape(1, -1)[0][0]

        except Exception as e:
            # Handle any errors that occur during form processing or prediction
            prediction_value = f"Error: {str(e)}"  # Assign the error message as a string

    # Render the template with the prediction value
    return render_template('yield.html', prediction=prediction_value, title="yield")
def load_model():
    try:
        with open('plant_disease_model.pkl', 'rb') as model_file:
            return pickle.load(model_file)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def process_image(image_data):
    try:
        # Convert image bytes to numpy array
        # Note: This is a simplified version - you'll need to adjust the image processing
        # to match your model's expected input format
        image_array = np.frombuffer(image_data.read(), np.uint8)
        # Reshape array to match model input shape (assuming 128x128 images)
        processed_array = image_array.reshape(1, 128, 128, 3)
        return processed_array
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

@app.route('/predict_disease/', methods=['GET', 'POST'])
def predict_disease():
    prediction_text = None
    
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                flash('No file uploaded', 'error')
                return redirect(request.url)
            
            file = request.files['file']
            
            if file.filename == '':
                flash('No file selected', 'error')
                return redirect(request.url)
            
            if file:
                # Process the image
                image_data = io.BytesIO(file.read())
                processed_image = process_image(image_data)
                
                if processed_image is None:
                    flash('Error processing image', 'error')
                    return redirect(request.url)
                
                # Load model and make prediction
                model = load_model()
                if model is None:
                    flash('Error loading model', 'error')
                    return redirect(request.url)
                
                # Make prediction
                result_index = np.argmax(model.predict(processed_image))
                
                # Class names list
                class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
                
                # Format prediction
                prediction = class_names[result_index]
                plant, condition = prediction.split('___')
                prediction_text = f"Plant: {plant}\nCondition: {condition.replace('_', ' ')}"
                flash(prediction_text, 'success')
                
        except Exception as e:
            flash(f'Error during prediction: {str(e)}', 'error')
    
    return render_template('disease.html', title="Plant Disease Prediction")

@app.route("/register",methods=['GET','POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        flash(f'Account created for {form.username.data}!', 'success')
        return redirect(url_for('index'))
    return render_template('register.html',title = 'Register',form = form)

@app.route("/login", methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():  # This checks if the form is submitted and validated
        if form.email.data == 'admin@blog.com' and form.password.data == 'password':
            flash('You have been logged in!', 'success')  # Corrected 'siccess' to 'success'
            return redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please Check username and Password', 'danger')  # Corrected 'Unsccessful' to 'Unsuccessful'
    return render_template('login.html', title='Login', form=form)






# Main block
if __name__ == "__main__":
    app.run(debug=True)
