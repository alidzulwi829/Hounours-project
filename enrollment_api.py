import cv2
import base64
import psycopg2
import numpy as np
import face_recognition
from io import BytesIO
from PIL import Image, UnidentifiedImageError
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Database connection details
DB_CONFIG = {
    "host": "localhost",
    "database": "recognition",
    "user": "millicent",
    "password": "milli@21"
}

# Function to encode an image to Base64
def image_to_base64(image):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) 
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')  

# Function to extract face encodings from an image
def get_face_encoding(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    face_locations = face_recognition.face_locations(rgb_image)  
    if not face_locations:
        print("No face found in the image.")
        return None  

    encodings = face_recognition.face_encodings(rgb_image, face_locations)
    return encodings[0] if encodings else None 

# Function to store user info in PostgreSQL database
def store_user_info(name, surname,  image_base64, face_encoding):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Convert face encoding to a binary format (BYTEA)
        face_encoding_bytes = np.array(face_encoding, dtype=np.float32).tobytes()

        insert_query = """
            INSERT INTO Information (name, surname,  image_base64, face_encoding)
            VALUES (%s, %s, %s, %s, %s, %s, %s);
        """
        cursor.execute(insert_query, (name, surname, image_base64, face_encoding_bytes))
        conn.commit()

        if cursor.rowcount > 0:
            print("User info successfully stored in the database.")
        else:
            print("Insert failed, no rows affected.")

        cursor.close()
        conn.close()
    except Exception as e:
        print("Database error:", e)
        if conn:
            conn.rollback()


@app.route('/enroll/details', methods=['POST'])
def collect_user_details():
    data = request.json
    name = data.get("name")
    surname = data.get("surname")

    # Add logging to verify data is received
    print("Received details:")
    print(f"Name: {name}")
    print(f"Surname: {surname}")

    if not name or not surname :
        return jsonify({"error": "Name, surname are required"}), 400

    return jsonify({"message": "Details received", "name": name, "surname": surname}), 200

# Step 2: API endpoint to process user's face image from frontend
@app.route('/enroll/capture', methods=['POST'])
def capture_user_face():
    data = request.json
    image_base64 = data.get('image_base64')

    if not image_base64:
        return jsonify({"error": "Image data is required"}), 400

    try:
        # Decode the base64 image
        image_data = base64.b64decode(image_base64)

        # Verify the length and type of the image data
        print(f"Decoded image size: {len(image_data)} bytes")
        print(f"Image type: {type(image_data)}")

        # Attempt to open the image
        image = np.array(Image.open(BytesIO(image_data)))

        # Extract face encoding
        face_encoding = get_face_encoding(image)
        if face_encoding is None:
            return jsonify({"error": "No face detected in the image"}), 400

        return jsonify({"message": "Image processed successfully", "image_base64": image_base64, "face_encoding": face_encoding.tolist()}), 200

    except UnidentifiedImageError:
        print("Error processing image: cannot identify image file.")
        return jsonify({"error": "Failed to process image: cannot identify image file"}), 500
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({"error": f"Failed to process image: {e}"}), 500
   
@app.route('/enroll/confirm', methods=['POST'])
def confirm_and_save_user_info():
    data = request.json
    name = data.get("name")
    surname = data.get("surname")
    image_base64 = data.get("image_base64")
    face_encoding = data.get("face_encoding")

    # Log the received data for confirmation
    print("Received data for confirmation:")
    print(f"Name: {name}")
    print(f"Surname: {surname}")
    print(f"Image Base64: {image_base64[:30]}...")  # Print only first 30 chars
    print(f"Face Encoding: {face_encoding}")

    if not name or not surname or not image_base64 or not face_encoding:
        return jsonify({"error": "Name, surname,  image, and face encoding are required"}), 400

    try:
        # Convert face encoding back to a NumPy array
        face_encoding_np = np.array(face_encoding, dtype=np.float32)
    except Exception as e:
        print(f"Error converting face encoding: {e}")
        return jsonify({"error": f"Invalid face encoding format: {e}"}), 400

    # Store the user information in the database
    try:
        store_user_info(name, surname, image_base64, face_encoding_np)
    except Exception as e:
        return jsonify({"error": f"Failed to save user info: {e}"}), 500

    return jsonify({"message": "User enrolled successfully"}), 200


# Step 4: API endpoint to get face encodings from an image
@app.route('/face_encodings', methods=['POST'])
def get_face_encodings_from_image():
    data = request.json
    image_base64 = data.get("image_bytes")

    if not image_base64:
        return jsonify({"error": "Image data is required"}), 400

    try:
        # Decode the base64 image
        image_data = base64.b64decode(image_base64)
        image = np.array(Image.open(BytesIO(image_data)))

        # Get face encodings
        face_encoding = get_face_encoding(image)
        if face_encoding is None:
            return jsonify({"error": "No face detected in the image"}), 400

        return jsonify({"encodings": face_encoding.tolist()}), 200

    except UnidentifiedImageError:
        print("Error processing image: cannot identify image file.")
        return jsonify({"error": "Failed to process image: cannot identify image file"}), 500
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({"error": f"Failed to process image: {e}"}), 500

# Run the Flask application
if __name__ == "__main__":
    app.run(debug=True, port=5003)


