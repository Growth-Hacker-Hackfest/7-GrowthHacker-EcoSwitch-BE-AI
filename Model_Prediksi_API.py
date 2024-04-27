from sklearn.preprocessing import OneHotEncoder
import uvicorn
from fastapi import FastAPI
import pandas as pd

app = FastAPI()

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

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)