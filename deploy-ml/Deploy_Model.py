from sklearn.preprocessing import OneHotEncoder
import uvicorn
import pandas as pd
from fastapi import FastAPI, File, UploadFile
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

app = FastAPI()

model = load_model('app/animall_person_other_v2_fine_tuned.h5')
class_names = ['animal', 'other', 'person']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    predictions = model.predict(x)
    predicted_class = class_names[np.argmax(predictions)]
    return {"class": predicted_class}

@app.post('/predict_kwh')
def predict_kwh(daya_listrik, jenis_alat, kulkas_num, kulkas_consume_hour, kulkas_power, ac_num, ac_consume_hour, ac_power, lamp_type, lamp_num, lamp_consume_hour, lamp_power):
    power = (kulkas_num * kulkas_consume_hour * kulkas_power) + (ac_num * ac_consume_hour * ac_power) + (lamp_num * lamp_consume_hour * lamp_power)
    return (power*30) / 1000

@app.post('/predict_price')
def predict_price(daya_listrik, jenis_alat, kulkas_inverter, kulkas_num, kulkas_consume_hour, kulkas_power, ac_inverter, ac_num, ac_consume_hour, ac_power, lamp_type, lamp_num, lamp_consume_hour, lamp_power):
    power_kwh = predict_kwh(daya_listrik, jenis_alat, kulkas_inverter, kulkas_num, kulkas_consume_hour, kulkas_power, ac_inverter, ac_num, ac_consume_hour, ac_power, lamp_type, lamp_num, lamp_consume_hour, lamp_power)

    if (daya_listrik == 450):
        power_kwh *= 415
    elif (daya_listrik == 900):
        power_kwh *= 1352
    elif (daya_listrik == 1300):
        power_kwh *= 1444
    elif (daya_listrik == 2200):
        power_kwh *= 1444
    else:
        power_kwh *= 1699
    return power_kwh

@app.post('/predict_co2')
def predict_co2(daya_listrik, jenis_alat, kulkas_num, kulkas_consume_hour, kulkas_power, ac_num, ac_consume_hour, ac_power, lamp_type, lamp_num, lamp_consume_hour, lamp_power):
    electric_carbon = predict_kwh(daya_listrik, jenis_alat, kulkas_num, kulkas_consume_hour, kulkas_power, ac_inverter, ac_num, ac_consume_hour, ac_power, lamp_type, lamp_num, lamp_consume_hour, lamp_power) * 0.0094

    carbon_ac_non_inverter= 0.10396
    carbon_kulkas_inverter= 0.06500
    carbon_kulkas_non_inverter= 0.06600

    kulkas_carbon = (kulkas_num / 24) * kulkas_consume_hour
    ac_carbon = ac_num * ac_consume_hour
    if(jenis_alat=="kulkas inverter"):
        kulkas_carbon *= carbon_kulkas_inverter
    elif(jenis_alat=="kulkas non inverter"):
        kulkas_carbon *= carbon_kulkas_non_inverter
    elif(jenis_alat=="ac non inverter"):
        ac_carbon *= carbon_ac_non_inverter

    lamp_carbon = lamp_consume_hour * lamp_num
    if (lamp_type=="pijar"):
        lamp_carbon *= 0.02150
    elif(lamp_type =="neon"):
        lamp_carbon*=0.00540
    else:
        lamp_carbon*=0.00240

    carbon = electric_carbon + lamp_carbon + ac_carbon + kulkas_carbon
    carbon *= 1000
    rounded_carbon_percentage = round((carbon / 12), 2)
    return rounded_carbon_percentage

@app.post('/rekomendasi')
def rekomendasi(daya_listrik, jenis_alat, kulkas_num, kulkas_consume_hour, kulkas_power, ac_num, ac_consume_hour, ac_power, lamp_type, lamp_num, lamp_consume_hour, lamp_power):
    if daya_listrik <= 450:
        rec_daya_listrik = 450
    elif daya_listrik <= 900:
        rec_daya_listrik = 900
    elif daya_listrik <= 1300:
        rec_daya_listrik = 1300
    elif daya_listrik <= 2200:
        rec_daya_listrik = 2200
    elif daya_listrik <= 3500:
        rec_daya_listrik = 3500
    elif daya_listrik <= 5500:
        rec_daya_listrik = 5500
    else:
        rec_daya_listrik = 6600

    if lamp_type == "pijar":
        rec_lamp_type = "neon"
    elif lamp_type == "neon":
        rec_lamp_type = "led"
    else:
        rec_lamp_type = "led"

    if jenis_alat == "kulkas inverter":
        rec_jenis_alat = "kulkas inverter"
    elif jenis_alat == "kulkas non inverter":
        rec_jenis_alat = "kulkas inverter"
    elif jenis_alat == "ac inverter":
        rec_jenis_alat = "ac inverter"
    else:
        rec_jenis_alat = "ac inverter"

    numeric = [kulkas_num, kulkas_consume_hour, kulkas_power, ac_num, ac_consume_hour, ac_power, lamp_num, lamp_consume_hour, lamp_power]
    rec_numeric = [round(num * 0.8) for num in numeric]

    output = "Berdasarkan pola penggunaan anda, saya sarankan untuk melakukan penghematan dengan mengurangi penggunaan alat elektronik menjadi:\n"
    output += f"daya_listrik: {rec_daya_listrik} Watt\n"
    output += f"jenis_alat: {rec_jenis_alat}\n"
    output += f"lamp_type: {rec_lamp_type}\n"

    parameter_names = ["kulkas_num", "kulkas_consume_hour", "kulkas_power", "ac_num", "ac_consume_hour", "ac_power", "lamp_num", "lamp_consume_hour", "lamp_power"]
    for i, name in enumerate(parameter_names):
        output += f"{name}: {rec_numeric[i]}\n"
    return output

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)