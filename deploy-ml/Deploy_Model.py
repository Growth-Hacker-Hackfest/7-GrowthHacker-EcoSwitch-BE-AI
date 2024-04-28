from sklearn.preprocessing import OneHotEncoder
import uvicorn
import pandas as pd
from fastapi import FastAPI, File, UploadFile
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import itertools
from pydantic import BaseModel

app = FastAPI()

model = load_model('human_detector.keras')
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
    return {"classification": predicted_class}


class PredictionRequest(BaseModel):
    daya: int
    key: list[str]

@app.post('/predict_combined')
def predict_combined(request: PredictionRequest):
    daya = request.daya
    key = request.key

    ac_inverter_arr = []
    ac_non_inverter_arr = []
    kulkas_inverter_arr = []
    kulkas_non_inverter_arr = []
    lampu_pijar_arr = []
    lampu_led_arr = []
    televisi_arr = []
    kipas_angin_arr = []

    for i in key:
        if isinstance(i, str):
            if "ac inverter" in i:
                ac_inverter_arr.append(i)
            elif "ac non inverter" in i:
                ac_non_inverter_arr.append(i)
            elif "kulkas inverter" in i:
                kulkas_inverter_arr.append(i)
            elif "kulkas non inverter" in i:
                kulkas_non_inverter_arr.append(i)
            elif "lampu pijar" in i:
                lampu_pijar_arr.append(i)
            elif "lampu led" in i:
                lampu_led_arr.append(i)
            elif "televisi" in i:
                televisi_arr.append(i)
            elif "kipas angin" in i:
                kipas_angin_arr.append(i)

    ac_inverter_data = [list(group) for _, group in itertools.groupby(key, lambda x: x in ac_inverter_arr) if group]
    ac_non_inverter_data = [list(group) for _, group in itertools.groupby(key, lambda x: x in ac_non_inverter_arr) if group]
    kulkas_inverter_data = [list(group) for _, group in itertools.groupby(key, lambda x: x in kulkas_inverter_arr) if group]
    kulkas_non_inverter_data = [list(group) for _, group in itertools.groupby(key, lambda x: x in kulkas_non_inverter_arr) if group]
    lampu_pijar_data = [list(group) for _, group in itertools.groupby(key, lambda x: x in lampu_pijar_arr) if group]
    lampu_led_data = [list(group) for _, group in itertools.groupby(key, lambda x: x in lampu_led_arr) if group]
    televisi_data = [list(group) for _, group in itertools.groupby(key, lambda x: x in televisi_arr) if group]
    kipas_angin_data = [list(group) for _, group in itertools.groupby(key, lambda x: x in kipas_angin_arr) if group]

    ac_inverter_num = ac_inverter_power = ac_inverter_consume_hour = 0
    ac_non_inverter_num = ac_non_inverter_power = ac_non_inverter_consume_hour = 0
    kulkas_inverter_num = kulkas_inverter_power = kulkas_inverter_consume_hour = 0
    kulkas_non_inverter_num = kulkas_non_inverter_power = kulkas_non_inverter_consume_hour = 0
    lampu_pijar_num = lampu_pijar_power = lampu_pijar_consume_hour = 0
    lampu_led_num = lampu_led_power = lampu_led_consume_hour = 0
    televisi_num = televisi_power = televisi_consume_hour = 0
    kipas_angin_num = kipas_angin_power = kipas_angin_consume_hour = 0

    for data in ac_inverter_data:
        ac_inverter_num, ac_inverter_power, ac_inverter_consume_hour = int(data[1]), int(data[2]), int(data[3])

    for data in ac_non_inverter_data:
        ac_non_inverter_num, ac_non_inverter_power, ac_non_inverter_consume_hour = int(data[1]), int(data[2]), int(data[3])

    for data in kulkas_inverter_data:
        kulkas_inverter_num, kulkas_inverter_power, kulkas_inverter_consume_hour = int(data[1]), int(data[2]), int(data[3])

    for data in kulkas_non_inverter_data:
        kulkas_non_inverter_num, kulkas_non_inverter_power, kulkas_non_inverter_consume_hour = int(data[1]), int(data[2]), int(data[3])

    for data in lampu_pijar_data:
        lampu_pijar_num, lampu_pijar_power, lampu_pijar_consume_hour = int(data[1]), int(data[2]), int(data[3])

    for data in lampu_led_data:
        lampu_led_num, lampu_led_power, lampu_led_consume_hour = int(data[1]), int(data[2]), int(data[3])

    for data in televisi_data:
        televisi_num, televisi_power, televisi_consume_hour = int(data[1]), int(data[2]), int(data[3])

    for data in kipas_angin_data:
        kipas_angin_num, kipas_angin_power, kipas_angin_consume_hour = int(data[1]), int(data[2]), int(data[3])

    power = (kulkas_inverter_num * kulkas_inverter_consume_hour * kulkas_inverter_power) + (ac_inverter_num * ac_inverter_consume_hour * ac_inverter_power) + (lampu_pijar_num * lampu_pijar_consume_hour * lampu_pijar_power) +(lampu_led_num * lampu_led_consume_hour * lampu_led_power)+ (televisi_num * televisi_consume_hour * televisi_power) + (kipas_angin_num * kipas_angin_consume_hour * kipas_angin_power)
    total_kwh = (power * 30) / 1000
    
    power_kwh = total_kwh

    if (daya == 450):
        power_kwh *= 415
    elif (daya == 900):
        power_kwh *= 1352
    elif (daya == 1300):
        power_kwh *= 1444
    elif (daya == 2200):
        power_kwh *= 1444
    else:
        power_kwh *= 1699
    price = power_kwh

    electric_carbon = total_kwh * 0.0094

    kulkas_inverter_carbon = (kulkas_inverter_num / 24) * kulkas_inverter_consume_hour
    ac_inverter_carbon = ac_inverter_num * ac_inverter_consume_hour
    lampu_pijar_carbon = lampu_pijar_consume_hour * lampu_pijar_num

    kulkas_non_inverter_carbon = (kulkas_non_inverter_num / 24) * kulkas_non_inverter_consume_hour
    ac_non_inverter_carbon = ac_non_inverter_num * ac_non_inverter_consume_hour
    lampu_led_carbon = lampu_led_consume_hour * lampu_led_num

    carbon = (
        (electric_carbon)
        + (lampu_pijar_carbon * 0.215)
        + (lampu_led_carbon * 0.215)
        + (ac_inverter_carbon * 0.1)
        + (ac_non_inverter_carbon * 0.1)
        + (kulkas_inverter_carbon * 0.8)
        + (kulkas_non_inverter_carbon * 0.8)
    )
    carbon *= 1000
    rounded_carbon_percentage = round((carbon / 12), 2)

    return total_kwh, price, rounded_carbon_percentage

@app.post('/rekomendasi')
def rekomendasi(daya, jenis_perangkat, kulkas_num, kulkas_consume_hour, kulkas_power, ac_num, ac_consume_hour, ac_power, lamp_num, lamp_consume_hour, lamp_power):
    rec_daya = daya
    if jenis_perangkat == "lampu pijar":
        rec_lamp_type = "led"
    elif jenis_perangkat == "lampu led":
        rec_lamp_type = "led"
    elif jenis_perangkat == "kulkas inverter":
        rec_jenis_perangkat = "kulkas inverter"
    elif jenis_perangkat == "kulkas non inverter":
        rec_jenis_perangkat = "kulkas inverter"
    elif jenis_perangkat == "ac inverter":
        rec_jenis_perangkat = "ac inverter"
    else:
        rec_jenis_perangkat = "ac inverter"

    numeric = [kulkas_num, kulkas_consume_hour, kulkas_power, ac_num, ac_consume_hour, ac_power, lamp_num, lamp_consume_hour, lamp_power]
    rec_numeric = [round(num * 0.8) for num in numeric]

    output = "Berdasarkan pola penggunaan anda, saya sarankan untuk melakukan penghematan dengan mengurangi penggunaan alat elektronik menjadi:\n"
    output += f"daya: {rec_daya} Watt\n"
    output += f"jenis_perangkat: {rec_jenis_perangkat}\n"
    output += f"lamp_type: {rec_lamp_type}\n"

    parameter_names = ["kulkas_num", "kulkas_consume_hour", "kulkas_power", "ac_num", "ac_consume_hour", "ac_power", "lamp_num", "lamp_consume_hour", "lamp_power"]
    for i, name in enumerate(parameter_names):
        output += f"{name}: {rec_numeric[i]}\n"
    return output

class RecommendationRequest(BaseModel):
    key: list[str]
@app.post('/rekomendasi_v2')
def rekomendasi(request: RecommendationRequest):
    key = request.key

    ac_inverter_arr = []
    ac_non_inverter_arr = []
    kulkas_inverter_arr = []
    kulkas_non_inverter_arr = []
    lampu_pijar_arr = []
    lampu_led_arr = []
    televisi_arr = []
    kipas_angin_arr = []

    for i in key:
        if isinstance(i, str):
            if "ac inverter" in i:
                ac_inverter_arr.append(i)
            elif "ac non inverter" in i:
                ac_non_inverter_arr.append(i)
            elif "kulkas inverter" in i:
                kulkas_inverter_arr.append(i)
            elif "kulkas non inverter" in i:
                kulkas_non_inverter_arr.append(i)
            elif "lampu pijar" in i:
                lampu_pijar_arr.append(i)
            elif "lampu led" in i:
                lampu_led_arr.append(i)
            elif "televisi" in i:
                televisi_arr.append(i)
            elif "kipas angin" in i:
                kipas_angin_arr.append(i)

    ac_inverter_data = [list(group) for _, group in itertools.groupby(key, lambda x: x in ac_inverter_arr) if group]
    ac_non_inverter_data = [list(group) for _, group in itertools.groupby(key, lambda x: x in ac_non_inverter_arr) if group]
    kulkas_inverter_data = [list(group) for _, group in itertools.groupby(key, lambda x: x in kulkas_inverter_arr) if group]
    kulkas_non_inverter_data = [list(group) for _, group in itertools.groupby(key, lambda x: x in kulkas_non_inverter_arr) if group]
    lampu_pijar_data = [list(group) for _, group in itertools.groupby(key, lambda x: x in lampu_pijar_arr) if group]
    lampu_led_data = [list(group) for _, group in itertools.groupby(key, lambda x: x in lampu_led_arr) if group]
    televisi_data = [list(group) for _, group in itertools.groupby(key, lambda x: x in televisi_arr) if group]
    kipas_angin_data = [list(group) for _, group in itertools.groupby(key, lambda x: x in kipas_angin_arr) if group]

    ac_inverter_num = ac_inverter_power = ac_inverter_consume_hour = 0
    ac_non_inverter_num = ac_non_inverter_power = ac_non_inverter_consume_hour = 0
    kulkas_inverter_num = kulkas_inverter_power = kulkas_inverter_consume_hour = 0
    kulkas_non_inverter_num = kulkas_non_inverter_power = kulkas_non_inverter_consume_hour = 0
    lampu_pijar_num = lampu_pijar_power = lampu_pijar_consume_hour = 0
    lampu_led_num = lampu_led_power = lampu_led_consume_hour = 0
    televisi_num = televisi_power = televisi_consume_hour = 0
    kipas_angin_num = kipas_angin_power = kipas_angin_consume_hour = 0

    for data in ac_inverter_data:
        ac_inverter_num, ac_inverter_power, ac_inverter_consume_hour = int(data[1]), int(data[2]), int(data[3])

    for data in ac_non_inverter_data:
        ac_non_inverter_num, ac_non_inverter_power, ac_non_inverter_consume_hour = int(data[1]), int(data[2]), int(data[3])

    for data in kulkas_inverter_data:
        kulkas_inverter_num, kulkas_inverter_power, kulkas_inverter_consume_hour = int(data[1]), int(data[2]), int(data[3])

    for data in kulkas_non_inverter_data:
        kulkas_non_inverter_num, kulkas_non_inverter_power, kulkas_non_inverter_consume_hour = int(data[1]), int(data[2]), int(data[3])

    for data in lampu_pijar_data:
        lampu_pijar_num, lampu_pijar_power, lampu_pijar_consume_hour = int(data[1]), int(data[2]), int(data[3])

    for data in lampu_led_data:
        lampu_led_num, lampu_led_power, lampu_led_consume_hour = int(data[1]), int(data[2]), int(data[3])

    for data in televisi_data:
        televisi_num, televisi_power, televisi_consume_hour = int(data[1]), int(data[2]), int(data[3])

    for data in kipas_angin_data:
        kipas_angin_num, kipas_angin_power, kipas_angin_consume_hour = int(data[1]), int(data[2]), int(data[3])
    
    import pandas as pd

    data = {
        'ac_inverter_num': [ac_inverter_num],
        'ac_inverter_power': [ac_inverter_power],
        'ac_inverter_consume_hour': [ac_inverter_consume_hour],
        'ac_non_inverter_num': [ac_non_inverter_num],
        'ac_non_inverter_power': [ac_non_inverter_power],
        'ac_non_inverter_consume_hour': [ac_non_inverter_consume_hour],
        'kulkas_inverter_num': [kulkas_inverter_num],
        'kulkas_inverter_power': [kulkas_inverter_power],
        'kulkas_inverter_consume_hour': [kulkas_inverter_consume_hour],
        'kulkas_non_inverter_num': [kulkas_non_inverter_num],
        'kulkas_non_inverter_power': [kulkas_non_inverter_power],
        'kulkas_non_inverter_consume_hour': [kulkas_non_inverter_consume_hour],
        'lampu_pijar_num': [lampu_pijar_num],
        'lampu_pijar_power': [lampu_pijar_power],
        'lampu_pijar_consume_hour': [lampu_pijar_consume_hour],
        'lampu_led_num': [lampu_led_num],
        'lampu_led_power': [lampu_led_power],
        'lampu_led_consume_hour': [lampu_led_consume_hour],
        'televisi_num': [televisi_num],
        'televisi_power': [televisi_power],
        'televisi_consume_hour': [televisi_consume_hour],
        'kipas_angin_num': [kipas_angin_num],
        'kipas_angin_power': [kipas_angin_power],
        'kipas_angin_consume_hour': [kipas_angin_consume_hour]
    }

    df = pd.DataFrame(data)

    if len(lampu_pijar_num) != 0:
        rec_lamp = "led"
    if len(kulkas_non_inverter_num) != 0:
        rec_kulkas = "kulkas non inverter"
    if len(ac_non_inverter_num) != 0:
        rec_ac = "ac non inverter"
    
    kulkas = kulkas_inverter_num+kulkas_non_inverter_num
    pow_inv_kulkas = kulkas_inverter_power
    pow_non_inv_kulkas = kulkas_non_inverter_power
    lampu = lampu_led_num+lampu_pijar_num
    pow_led = lampu_led_power
    pow_pijar = lampu_pijar_power
    ac = ac_inverter_num+ac_non_inverter_num
    pow_inv_ac = ac_inverter_power
    pow_non_inv_ac = ac_non_inverter_power
    numeric = [kulkas,lampu,ac,pow_inv_kulkas,pow_non_inv_kulkas,pow_led,pow_pijar,pow_inv_ac,pow_non_inv_ac]
    rec_numeric = [round(num * 0.8) for num in numeric]

    output = "Berdasarkan pola penggunaan anda, saya sarankan untuk melakukan penghematan dengan mengurangi penggunaan alat elektronik menjadi:\n"
    output += "rekomendasi jenis lampu: "+rec_lamp 
    output += "rekomendasi jenis lampu: "+rec_kulkas 
    output += "rekomendasi jenis ac: "+rec_ac 

    parameter_names = ["kulkas","lampu","ac","pow_inv_kulkas","pow_non_inv_kulkas","pow_led","pow_pijar","pow_inv_ac","pow_non_inv_ac"]
    for i, name in enumerate(parameter_names):
        output += f"{name}: {rec_numeric[i]}\n"
    return output

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
