from pathlib import Path
import os

from io import BytesIO

import json

import base64



import numpy as np

from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException

from fastapi.middleware.cors import CORSMiddleware



import tensorflow as tf

from tensorflow.keras import layers, models

from tensorflow.keras.applications import MobileNetV2

import cv2



# Ultralytics is imported lazily to reduce Render startup memory.
YOLO = None
# ============================================================

# PATHS

# ============================================================



BASE_DIR = Path(__file__).resolve().parent



MODEL_DIR = BASE_DIR / "model"



WEIGHTS_PATH = MODEL_DIR / "mobilenetv2_best.weights.h5"

CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"





# ============================================================

# CLASS NAMES

# ============================================================



DEFAULT_CLASSES = [

    "Bud_Borer",

    "Healthy_Foot",

    "Healthy_Leaf",

    "Healthy_Nut",

    "Healthy_Trunk",

    "Mahali_Koleroga",

    "Stem_Bleeding",

    "Stem_Cracking",

    "Yellow_Leaf_Disease",

]



if CLASS_NAMES_PATH.exists():

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:

        CLASS_NAMES = json.load(f)

else:

    CLASS_NAMES = DEFAULT_CLASSES





# ============================================================

# HEALTHY / DISEASE CLASSES

# ============================================================



HEALTHY_CLASSES = {

    "Healthy_Foot",

    "Healthy_Leaf",

    "Healthy_Nut",

    "Healthy_Trunk",

}



# Final valid deployed model evaluation from the training run

MODEL_TEST_ACCURACY = 85.90

MODEL_TEST_LOSS = 0.5225

MODEL_INPUT_SIZE = "224x224"

MODEL_VERSION = "ArecaCare MobileNetV2 - 9 Class"



YOLO_GUARD = None

YOLO_GUARD_ERROR = None

YOLO_GUARD_MODEL = "yolo11n.pt"

COCO_REJECT_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck", "boat", "airplane", "train", "stop sign", "traffic light", "backpack", "handbag", "suitcase", "chair", "couch", "bed", "laptop", "cell phone", "tv", "remote", "keyboard", "mouse", "bottle", "cup", "book", "clock", "sports ball", "skateboard", "surfboard", "tennis racket", "frisbee", "umbrella", "tie"}





# ============================================================

# ADVISORY

# ============================================================



ADVISORY = {

    "Healthy_Foot":

        "The plant foot appears healthy. Continue proper drainage, field sanitation and regular monitoring.",



    "Healthy_Leaf":

        "The leaf appears healthy. Continue regular monitoring, balanced nutrition and good field sanitation.",



    "Healthy_Nut":

        "The nut appears healthy. Continue regular crop monitoring and proper irrigation management.",



    "Healthy_Trunk":

        "The trunk appears healthy. Continue monitoring for cracks, bleeding symptoms and pest activity.",



    "Bud_Borer":

        "Bud borer symptoms detected. Inspect the central spindle and growing point. Remove severely affected plant parts and follow locally recommended pest management practices.",



    "Mahali_Koleroga":

        "Possible Mahali Koleroga detected. Check for water-soaked lesions and nut shedding. Improve field drainage and follow locally recommended fungicide management.",



    "Stem_Bleeding":

        "Possible stem bleeding detected. Inspect the trunk for dark bleeding patches. Remove affected material where appropriate and follow agricultural disease-management recommendations.",



    "Stem_Cracking":

        "Possible stem cracking detected. Inspect the trunk carefully and monitor crack progression. Maintain proper field management and seek local agricultural guidance.",



    "Yellow_Leaf_Disease":

        "Possible yellow leaf disease detected. Check nutrient status, root health and field conditions. Confirm the diagnosis with an agricultural expert before treatment.",

}





# ============================================================

# BUILD MOBILE NET V2 MODEL

# ============================================================



def build_model():



    base_model = MobileNetV2(

        input_shape=(224, 224, 3),

        include_top=False,

        weights=None,

    )



    base_model.trainable = True



    inputs = layers.Input(shape=(224, 224, 3))



    x = layers.RandomFlip("horizontal")(inputs)

    x = layers.RandomRotation(0.15)(x)

    x = layers.RandomZoom(0.15)(x)

    x = layers.RandomContrast(0.10)(x)



    # Same preprocessing used during training

    x = layers.Rescaling(

        1.0 / 127.5,

        offset=-1

    )(x)



    x = base_model(x, training=False)



    x = layers.GlobalAveragePooling2D()(x)



    x = layers.Dropout(0.35)(x)



    x = layers.Dense(

        128,

        activation="relu"

    )(x)



    x = layers.Dropout(0.25)(x)



    outputs = layers.Dense(

        len(CLASS_NAMES),

        activation="softmax"

    )(x)



    return models.Model(inputs, outputs)





# ============================================================

# LOAD MODEL

# ============================================================



MODEL = None

MODEL_ERROR = None





def load_model():



    global MODEL

    global MODEL_ERROR



    try:



        if not WEIGHTS_PATH.exists():

            raise FileNotFoundError(

                f"Model weights not found: {WEIGHTS_PATH}"

            )



        MODEL = build_model()

        MODEL.load_weights(WEIGHTS_PATH)

        MODEL_ERROR = None



        print("MobileNetV2 model loaded successfully.")



    except Exception as e:



        MODEL = None

        MODEL_ERROR = str(e)



        print("MODEL LOAD ERROR:")

        print(MODEL_ERROR)





# MobileNetV2 is loaded on demand after FastAPI starts.
# ============================================================

# YOLO INPUT OBJECT GUARD

# ============================================================



def load_input_guard_model():

    global YOLO_GUARD, YOLO_GUARD_ERROR

    if YOLO is None:

        YOLO_GUARD_ERROR = "Ultralytics is not installed. Run: pip install ultralytics"

        print("INPUT GUARD WARNING:", YOLO_GUARD_ERROR)

        return

    try:

        YOLO_GUARD = YOLO(YOLO_GUARD_MODEL)

        YOLO_GUARD_ERROR = None

        print("YOLO input-domain guard loaded successfully.")

    except Exception as e:

        YOLO_GUARD = None

        YOLO_GUARD_ERROR = str(e)

        print("YOLO INPUT GUARD WARNING:", e)



# YOLO is loaded on demand when the guard is enabled.
# ============================================================

# IMAGE DOMAIN VALIDATION

# ============================================================

# The 9-class disease model is a closed-set classifier. Without a

# separate input guard, a non-plant image (for example a scooter

# advertisement) would still be forced into one of the 9 classes.

# This lightweight guard checks whether the image has enough

# vegetation/plant-like colour and texture before classification.

# It is an input-domain filter, NOT a second disease classifier.






# ============================================================
# LAZY AI LOADING / RENDER MEMORY CONTROL
# ============================================================

IS_RENDER = os.getenv("RENDER", "").lower() == "true"

# Locally, keep the existing YOLO input guard enabled by default.
# On Render Free, disable it by default because TensorFlow + YOLO/PyTorch
# can exceed the free instance memory limit.
ENABLE_YOLO_GUARD = os.getenv(
    "ENABLE_YOLO_GUARD",
    "false" if IS_RENDER else "true"
).lower() == "true"


def ensure_model_loaded():
    if MODEL is None:
        print("Loading MobileNetV2 model on demand...")
        load_model()

    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded: {MODEL_ERROR}"
        )


def ensure_input_guard_loaded():
    global YOLO_GUARD, YOLO_GUARD_ERROR, YOLO

    if not ENABLE_YOLO_GUARD or YOLO_GUARD is not None:
        return

    if YOLO is None:
        try:
            from ultralytics import YOLO as YOLOClass
            YOLO = YOLOClass
        except Exception as e:
            YOLO_GUARD_ERROR = f"Ultralytics import failed: {e}"
            print("INPUT GUARD WARNING:", YOLO_GUARD_ERROR)
            return

    print("Loading YOLO input guard on demand...")
    load_input_guard_model()


def release_yolo_guard():
    global YOLO_GUARD

    if ENABLE_YOLO_GUARD and YOLO_GUARD is not None:
        YOLO_GUARD = None
        try:
            import gc
            gc.collect()
        except Exception:
            pass


def validate_plant_image(image: Image.Image):



    small = image.convert("RGB").resize((224, 224))

    rgb = np.asarray(small, dtype=np.uint8)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)



    h = hsv[:, :, 0]

    sat = hsv[:, :, 1]

    value = hsv[:, :, 2]



    # Green vegetation: leaves / shoots / healthy plant tissue.

    green = (

        (h >= 25) &

        (h <= 100) &

        (sat >= 35) &

        (value >= 35)

    )



    # Yellow tissue: useful for yellow-leaf disease images.

    yellow = (

        (h >= 15) &

        (h < 35) &

        (sat >= 45) &

        (value >= 45)

    )



    # Brown/dark plant material: trunk, stem, dried/necrotic tissue.

    brown = (

        (h >= 5) &

        (h < 25) &

        (sat >= 45) &

        (value >= 25) &

        (value < 190)

    )



    plant_mask = green | yellow | brown

    plant_ratio = float(np.mean(plant_mask))



    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray, 60, 160)

    edge_ratio = float(np.mean(edges > 0))



    # ------------------------------------------------------------

    # HUMAN-FACE GUARD

    # ------------------------------------------------------------

    # A face photo can contain enough skin/yellow/brown pixels to pass

    # the simple vegetation-colour test. Reject prominent human faces

    # before the closed-set disease classifier sees the image.

    # This is intentionally conservative: the app is designed for

    # clear plant-part images, not portraits.

    face_count = 0

    largest_face_ratio = 0.0



    try:

        face_cascade = cv2.CascadeClassifier(

            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

        )



        if not face_cascade.empty():

            faces = face_cascade.detectMultiScale(

                gray,

                scaleFactor=1.03,

                minNeighbors=3,

                minSize=(24, 24),

            )



            face_count = len(faces)



            image_area = float(rgb.shape[0] * rgb.shape[1])



            if face_count:

                largest_face_ratio = max(

                    ((w * h) / image_area) for (x, y, w, h) in faces

                )



    except Exception:

        # If OpenCV face detection is unavailable, continue with the

        # existing plant/poster checks rather than crashing the API.

        face_count = 0

        largest_face_ratio = 0.0



    # ------------------------------------------------------------

    # YOLO OBJECT GUARD

    # ------------------------------------------------------------

    detected_objects = []

    reject_objects = []

    largest_reject_conf = 0.0



    if YOLO_GUARD is not None:

        try:

            results = YOLO_GUARD.predict(

                source=np.asarray(image.convert("RGB")),

                imgsz=320,

                conf=0.30,

                iou=0.45,

                verbose=False,

            )

            if results:

                result = results[0]

                names = result.names

                if result.boxes is not None:

                    for cls_id, conf_score in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()):

                        object_name = str(names[int(cls_id)]).lower()

                        score = float(conf_score)

                        detected_objects.append({"class": object_name, "confidence": round(score * 100, 2)})

                        if object_name in COCO_REJECT_CLASSES and score >= 0.30:

                            reject_objects.append(object_name)

                            largest_reject_conf = max(largest_reject_conf, score)

        except Exception as e:

            print("YOLO input guard inference warning:", e)



    # Large white/near-white backgrounds combined with very little

    # plant-like content are typical of posters, documents and ads.

    bright_background = (

        (sat < 35) &

        (value > 190)

    )

    bright_ratio = float(np.mean(bright_background))



    # Main threshold chosen to reject obvious non-plant uploads while

    # allowing green, yellow and brown arecanut tissue.

    plant_like = plant_ratio >= 0.10



    # Additional poster/background guard. This catches images such as

    # the white/red scooter advertisement without blocking ordinary

    # close-up plant photos that contain enough plant pixels.

    poster_like = (

        plant_ratio < 0.13 and

        bright_ratio > 0.42 and

        edge_ratio > 0.025

    )



    # A prominent detected face is a strong indication that the upload

    # is a portrait/person photo rather than a clear plant-part image.

    face_like = face_count > 0 and largest_face_ratio >= 0.008

    object_like = len(reject_objects) > 0



    valid = bool(

        plant_like

        and not poster_like

        and not face_like

        and not object_like

    )



    if valid:

        message = "Plant-like image accepted for ArecaCare analysis."

    elif object_like:

        object_text = ", ".join(sorted(set(reject_objects)))

        message = (

            f"Non-plant object detected ({object_text}). Please upload a "

            "clear photo of the arecanut leaf, nut, trunk, bud or whole "

            "arecanut plant."

        )

    elif face_like:

        message = (

            "Human face detected. Please upload a clear photo of the "

            "arecanut leaf, nut, trunk, bud or whole arecanut plant. "

            "Do not upload a portrait or person photo."

        )

    else:

        message = (

            "This image does not appear to contain enough visible plant "

            "material for ArecaCare analysis. Please upload a clear photo "

            "of an arecanut leaf, nut, trunk, bud or whole arecanut plant."

        )



    return {

        "valid": valid,

        "plant_ratio": round(plant_ratio * 100, 2),

        "edge_ratio": round(edge_ratio * 100, 2),

        "bright_background_ratio": round(bright_ratio * 100, 2),

        "face_count": int(face_count),

        "largest_face_ratio": round(largest_face_ratio * 100, 2),

        "detected_objects": detected_objects,

        "rejected_objects": sorted(set(reject_objects)),

        "largest_reject_confidence": round(largest_reject_conf * 100, 2),

        "yolo_guard_loaded": YOLO_GUARD is not None,

        "message": message,

    }





# ============================================================

# GRAD-CAM

# ============================================================



def generate_gradcam(image: Image.Image):



    if MODEL is None:

        raise RuntimeError(

            f"Model not loaded: {MODEL_ERROR}"

        )



    original_image = image.convert("RGB")

    resized_image = original_image.resize((224, 224))



    image_array = np.array(

        resized_image,

        dtype=np.float32

    )

    image_array = np.expand_dims(image_array, axis=0)



    base_model = None



    for layer in MODEL.layers:

        if isinstance(layer, tf.keras.Model):

            if "mobilenetv2" in layer.name.lower():

                base_model = layer

                break



    if base_model is None:

        raise RuntimeError("MobileNetV2 backbone not found.")



    target_layer = None



    for layer in reversed(base_model.layers):

        try:

            if len(layer.output.shape) == 4:

                target_layer = layer

                break

        except Exception:

            continue



    if target_layer is None:

        raise RuntimeError(

            "Could not find convolutional feature layer."

        )



    feature_model = tf.keras.Model(

        inputs=base_model.input,

        outputs=[

            target_layer.output,

            base_model.output

        ]

    )



    gap_layer = None

    dense_layers = []

    dropout_layers = []



    for layer in MODEL.layers:



        if isinstance(layer, layers.GlobalAveragePooling2D):

            gap_layer = layer



        elif isinstance(layer, layers.Dense):

            dense_layers.append(layer)



        elif isinstance(layer, layers.Dropout):

            dropout_layers.append(layer)



    if gap_layer is None:

        raise RuntimeError(

            "GlobalAveragePooling2D layer not found."

        )



    if len(dense_layers) < 2:

        raise RuntimeError(

            "Classification Dense layers not found."

        )



    if len(dropout_layers) < 2:

        raise RuntimeError(

            "Classification Dropout layers not found."

        )



    with tf.GradientTape() as tape:



        x = tf.cast(image_array, tf.float32)

        x = x / 127.5 - 1.0



        conv_outputs, backbone_output = feature_model(

            x,

            training=False

        )



        x = gap_layer(backbone_output)



        x = dropout_layers[0](

            x,

            training=False

        )



        x = dense_layers[0](x)



        x = dropout_layers[1](

            x,

            training=False

        )



        predictions = dense_layers[1](x)



        predicted_index = tf.argmax(predictions[0])



        class_score = predictions[

            0,

            predicted_index

        ]



    gradients = tape.gradient(

        class_score,

        conv_outputs

    )



    if gradients is None:

        raise RuntimeError(

            "Could not calculate Grad-CAM gradients."

        )



    pooled_gradients = tf.reduce_mean(

        gradients,

        axis=(0, 1, 2)

    )



    conv_outputs = conv_outputs[0]



    heatmap = tf.reduce_sum(

        conv_outputs * pooled_gradients,

        axis=-1

    )



    heatmap = tf.maximum(heatmap, 0)



    max_value = tf.reduce_max(heatmap)



    if float(max_value) > 0:

        heatmap = heatmap / max_value



    heatmap = heatmap.numpy()



    import cv2



    heatmap_uint8 = np.uint8(heatmap * 255)



    heatmap_resized = cv2.resize(

        heatmap_uint8,

        (

            original_image.width,

            original_image.height

        )

    )



    colored_heatmap = cv2.applyColorMap(

        heatmap_resized,

        cv2.COLORMAP_JET

    )



    colored_heatmap = cv2.cvtColor(

        colored_heatmap,

        cv2.COLOR_BGR2RGB

    )



    original_array = np.array(original_image)



    overlay = (

        0.55 * original_array +

        0.45 * colored_heatmap

    )



    overlay = np.clip(

        overlay,

        0,

        255

    ).astype(np.uint8)



    result_image = Image.fromarray(overlay)



    buffer = BytesIO()



    result_image.save(

        buffer,

        format="JPEG",

        quality=90

    )



    encoded_image = base64.b64encode(

        buffer.getvalue()

    ).decode("utf-8")



    predicted_index = int(predicted_index.numpy())



    confidence = float(

        predictions[0, predicted_index].numpy()

    )



    return {

        "image": f"data:image/jpeg;base64,{encoded_image}",

        "class_index": predicted_index,

        "confidence": confidence * 100,

        "layer": target_layer.name,

    }





# ============================================================

# FASTAPI

# ============================================================



app = FastAPI(

    title="ArecaCare AI API",

    version="2.0.0"

)





# ============================================================

# CORS

# ============================================================



app.add_middleware(

    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        *([os.getenv("FRONTEND_URL")] if os.getenv("FRONTEND_URL") else []),
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],

)





# ============================================================

# PING

# ============================================================



@app.get("/ping")

def ping():



    return {

        "status": "ok",

        "model_loaded": MODEL is not None,

        "model_error": MODEL_ERROR,

        "model": MODEL_VERSION,

        "model_test_accuracy": MODEL_TEST_ACCURACY,

        "model_test_loss": MODEL_TEST_LOSS,

        "classes": CLASS_NAMES,

        "output_classes": len(CLASS_NAMES),

        "input_validation": "plant + face + optional YOLO object domain guard",

        "yolo_guard_loaded": YOLO_GUARD is not None,

        "yolo_guard_model": YOLO_GUARD_MODEL,

    }





# ============================================================

# PREDICT

# ============================================================




@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "model_error": MODEL_ERROR,
        "yolo_guard_enabled": ENABLE_YOLO_GUARD,
        "yolo_guard_loaded": YOLO_GUARD is not None,
    }


@app.post("/api/predict")

async def predict(file: UploadFile = File(...)):

    ensure_input_guard_loaded()
    ensure_model_loaded()



    try:

        contents = await file.read()



        image = Image.open(

            BytesIO(contents)

        ).convert("RGB")



        validation = validate_plant_image(image)



        
        # Free YOLO/PyTorch memory before TensorFlow work.
        release_yolo_guard()
        if not validation["valid"]:

            raise HTTPException(

                status_code=422,

                detail={

                    "code": "INVALID_IMAGE",

                    "message": validation["message"],

                    "valid_image": False,

                    "validation": validation,

                },

            )



        image = image.resize((224, 224))



        image_array = np.array(

            image,

            dtype=np.float32

        )



        image_array = np.expand_dims(

            image_array,

            axis=0

        )



        # The model already contains the Rescaling layer.

        predictions = MODEL.predict(

            image_array,

            verbose=0

        )[0]



        class_index = int(

            np.argmax(predictions)

        )



        confidence = float(

            predictions[class_index]

        )



        predicted_class = CLASS_NAMES[

            class_index

        ]



        # --------------------------------------------------------

        # FINAL CLASSIFICATION

        # --------------------------------------------------------

        # The application always reports the model's highest-scoring

        # trained class. There is intentionally no "Uncertain" UI state.

        # Per-image confidence and the independently evaluated model

        # test accuracy are reported separately.



        confidence_gap = 0.0

        if len(predictions) >= 2:

            sorted_scores = np.sort(predictions)

            confidence_gap = float(sorted_scores[-1] - sorted_scores[-2])



        if predicted_class in HEALTHY_CLASSES:

            status = "Healthy"

            advisory = ADVISORY.get(

                predicted_class,

                "Please continue regular monitoring."

            )

        else:

            status = "Diseased"

            advisory = ADVISORY.get(

                predicted_class,

                "Please consult an agricultural expert for confirmation."

            )



        # --------------------------------------------------------

        # TOP 3 PREDICTIONS

        # --------------------------------------------------------



        top_indices = np.argsort(

            predictions

        )[-3:][::-1]



        top_predictions = []



        for index in top_indices:

            top_predictions.append({

                "class": CLASS_NAMES[int(index)],

                "confidence": round(

                    float(predictions[index]) * 100,

                    2

                )

            })



        return {

            "success": True,

            "valid_image": True,

            "model": MODEL_VERSION,

            "model_test_accuracy": MODEL_TEST_ACCURACY,

            "model_test_loss": MODEL_TEST_LOSS,

            "class": predicted_class,

            "class_index": class_index,

            "confidence": round(

                confidence * 100,

                2

            ),

            "status": status,

            "advisory": advisory,

            "top_predictions": top_predictions,

            "confidence_gap": round(

                confidence_gap * 100,

                2

            ),

            "image_size": MODEL_INPUT_SIZE,

            "output_classes": len(CLASS_NAMES),

            "valid_image": True,

            "validation": validation,

            "evaluation": {

                "test_accuracy": MODEL_TEST_ACCURACY,

                "test_loss": MODEL_TEST_LOSS,

                "trained_classes": len(CLASS_NAMES),

            },

        }



    except Exception as e:

        raise HTTPException(

            status_code=400,

            detail=f"Prediction failed: {str(e)}"

        )





# ============================================================

# GRAD-CAM API

# ============================================================



@app.post("/api/gradcam")

async def gradcam(file: UploadFile = File(...)):

    ensure_input_guard_loaded()
    ensure_model_loaded()



    try:



        contents = await file.read()



        image = Image.open(

            BytesIO(contents)

        ).convert("RGB")



        validation = validate_plant_image(image)



        
        # Free YOLO/PyTorch memory before TensorFlow work.
        release_yolo_guard()
        if not validation["valid"]:

            raise HTTPException(

                status_code=422,

                detail=validation["message"]

            )



        result = generate_gradcam(image)



        predicted_index = result["class_index"]



        predicted_class = CLASS_NAMES[

            predicted_index

        ]



        return {

            "success": True,

            "model": "ArecaCare MobileNetV2",

            "class": predicted_class,

            "class_index": predicted_index,

            "confidence": round(

                result["confidence"],

                2

            ),

            "layer": result["layer"],

            "gradcam_image": result["image"],

        }
    except Exception as e:
        raise HTTPException(

            status_code=400,

            detail=f"Grad-CAM failed: {str(e)}"

        )
