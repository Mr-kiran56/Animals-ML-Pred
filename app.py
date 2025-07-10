from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os
import traceback
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# REAL ANIMAL NAME MAPPING
REAL_NAME_MAPPING = {
    "Animal_1": "Antelope",
    "Animal_2": "Badger",
    "Animal_3": "Bat",
    "Animal_4": "Bear",
    "Animal_5": "Bee",
    "Animal_6": "Beetle",
    "Animal_7": "Bison",
    "Animal_8": "Boar",
    "Animal_9": "Butterfly",
    "Animal_10": "Cat",
    "Animal_11": "Caterpillar",
    "Animal_12": "Chimpanzee",
    "Animal_13": "Cockroach",
    "Animal_14": "Cow",
    "Animal_15": "Coyote",
    "Animal_16": "Crab",
    "Animal_17": "Crow",
    "Animal_18": "Deer",
    "Animal_19": "Dog",
    "Animal_20": "Dolphin",
    "Animal_21": "Donkey",
    "Animal_22": "Dragonfly",
    "Animal_23": "Duck",
    "Animal_24": "Eagle",
    "Animal_25": "Elephant",
    "Animal_26": "Flamingo",
    "Animal_27": "Fly",
    "Animal_28": "Fox",
    "Animal_29": "Goat",
    "Animal_30": "Goldfish",
    "Animal_31": "Goose",
    "Animal_32": "Gorilla",
    "Animal_33": "Grasshopper",
    "Animal_34": "Hamster",
    "Animal_35": "Hare",
    "Animal_36": "Hedgehog",
    "Animal_37": "Hippopotamus",
    "Animal_38": "Hornbill",
    "Animal_39": "Horse",
    "Animal_40": "Hummingbird",
    "Animal_41": "Hyena",
    "Animal_42": "Jellyfish",
    "Animal_43": "Kangaroo",
    "Animal_44": "Koala",
    "Animal_45": "Ladybugs",
    "Animal_46": "Leopard",
    "Animal_47": "Lion",
    "Animal_48": "Lizard",
    "Animal_49": "Lobster",
    "Animal_50": "Mosquito",
    "Animal_51": "Moth",
    "Animal_52": "Mouse",
    "Animal_53": "Octopus",
    "Animal_54": "Okapi",
    "Animal_55": "Orangutan",
    "Animal_56": "Otter",
    "Animal_57": "Owl",
    "Animal_58": "Ox",
    "Animal_59": "Oyster",
    "Animal_60": "Panda",
    "Animal_61": "Parrot",
    "Animal_62": "Pelecaniformes",
    "Animal_63": "Penguin",
    "Animal_64": "Pig",
    "Animal_65": "Pigeon",
    "Animal_66": "Porcupine",
    "Animal_67": "Possum",
    "Animal_68": "Raccoon",
    "Animal_69": "Rat",
    "Animal_70": "Reindeer",
    "Animal_71": "Rhinoceros",
    "Animal_72": "Sandpiper",
    "Animal_73": "Seahorse",
    "Animal_74": "Seal",
    "Animal_75": "Shark",
    "Animal_76": "Sheep",
    "Animal_77": "Snake",
    "Animal_78": "Sparrow",
    "Animal_79": "Squid",
    "Animal_80": "Squirrel",
    "Animal_81": "Starfish",
    "Animal_82": "Swan",
    "Animal_83": "Tiger",
    "Animal_84": "Turkey",
    "Animal_85": "Turtle",
    "Animal_86": "Whale",
    "Animal_87": "Wolf",
    "Animal_88": "Wombat",
    "Animal_89": "Woodpecker",
    "Animal_90": "Zebra"
}

try:
    # Load model
    model = load_model('model.keras')
    logger.info("Model loaded successfully!")
    
    # Get model details
    if len(model.input_shape) == 4:
        input_shape = model.input_shape[1:3]  # Get (height, width)
    else:
        input_shape = model.input_shape[1:-1]  # Alternative for different architectures
    
    num_classes = model.output_shape[-1]  # Get class count
    logger.info(f"Model input shape: {input_shape}")
    logger.info(f"Number of output classes: {num_classes}")
    
    # Create class list based on the model's output size
    CLASSES = [f"Animal_{i}" for i in range(1, num_classes + 1)]
    
    # Map to real names
    DISPLAY_NAMES = [REAL_NAME_MAPPING.get(cls, cls) for cls in CLASSES]
    
    # Create animal image mapping
    ANIMAL_IMAGES = {}
    for cls in CLASSES:
        animal_name = REAL_NAME_MAPPING.get(cls, "animal")
        safe_name = animal_name.replace(' ', '_').lower()
        ANIMAL_IMAGES[cls] = f"static/{safe_name}.jpg"
    
    logger.info(f"Using {len(CLASSES)} classes")

except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    traceback.print_exc()
    input_shape = (244, 244)
    CLASSES = []
    DISPLAY_NAMES = []
    ANIMAL_IMAGES = {}
    model = None

@app.route('/')
def home():
    if not CLASSES or not model:
        return "Model failed to load. Check server logs.", 500
        
    return render_template('index.html', 
                           ANIMAL_CLASSES=DISPLAY_NAMES,
                           ANIMAL_IMAGES=ANIMAL_IMAGES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not model:
            return jsonify({'error': 'Model not loaded'}), 500
            
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Create uploads directory
        os.makedirs('uploads', exist_ok=True)
        img_path = os.path.join('uploads', file.filename)
        file.save(img_path)
        logger.info(f"Saved image to: {img_path}")
        
        # Preprocess image using model's input shape
        img = image.load_img(img_path, target_size=input_shape)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Make prediction
        logger.info("Making prediction...")
        prediction = model.predict(img_array)
        logger.info(f"Prediction shape: {prediction.shape}")
        logger.info(f"Prediction sample: {prediction[0][:5]}")
        
        predicted_index = np.argmax(prediction[0])
        logger.info(f"Predicted index: {predicted_index}")
        
        # Get top predictions with confidence
        top_indices = np.argsort(prediction[0])[::-1][:3]
        predictions = []
        
        for i in top_indices:
            # Classes are 1-indexed: Animal_1 to Animal_90
            animal_id = f"Animal_{i+1}"
            animal_name = REAL_NAME_MAPPING.get(animal_id, animal_id)
            confidence = float(prediction[0][i] * 100)
            
            predictions.append({
                'id': animal_id,
                'animal': animal_name,
                'confidence': confidence
            })
        
        # Find the highest confidence prediction
        main_prediction = predictions[0]
        
        # Clean up
        if os.path.exists(img_path):
            os.remove(img_path)
        
        return jsonify({
            'prediction': main_prediction['animal'],
            'confidence': main_prediction['confidence'],
            'predictions': predictions
        })
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)