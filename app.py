from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

with open("model (1).pkl", "rb") as f:
    model = pickle.load(f)

with open("preprocess.pkl", "rb") as f:
    preprocess = pickle.load(f)

median_values = preprocess.get("median_values", {})
mean_values = preprocess.get("mean_values", {})
mode_values = preprocess.get("mode_values", {})
label_encoders = preprocess.get("label_encoders", {})

def to_float_or_default(val, default):
    try:
        if val is None or val == '':
            return default
        return float(val)
    except:
        return default

def encode_col(col, raw_val):

    if raw_val is None or raw_val == '':
        return int(mode_values.get(col, 0))
    try:
        return int(raw_val)
    except:
        return int(mode_values.get(col, 0))

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":

        age = to_float_or_default(request.form.get("age"), median_values.get("age", 50))
        bp  = to_float_or_default(request.form.get("bp"), median_values.get("bp", 80))
        sg  = to_float_or_default(request.form.get("sg"), mean_values.get("sg", 1.015))
        al  = to_float_or_default(request.form.get("al"), median_values.get("al", 0))
        su  = to_float_or_default(request.form.get("su"), median_values.get("su", 0))
        bgr = to_float_or_default(request.form.get("bgr"), median_values.get("bgr", 100))
        bu  = to_float_or_default(request.form.get("bu"), median_values.get("bu", 40))
        sc  = to_float_or_default(request.form.get("sc"), median_values.get("sc", 1.2))
        sod = to_float_or_default(request.form.get("sod"), mean_values.get("sod", 138))
        pot = to_float_or_default(request.form.get("pot"), mean_values.get("pot", 4.2))
        hemo= to_float_or_default(request.form.get("hemo"), median_values.get("hemo", 13.0))
        pcv = to_float_or_default(request.form.get("pcv"), median_values.get("pcv", 40))
        wc  = to_float_or_default(request.form.get("wc"), median_values.get("wc", 8000))
        rc  = to_float_or_default(request.form.get("rc"), median_values.get("rc", 4.5))
        BMI = to_float_or_default(request.form.get("BMI"), 25.0)

 
        if BMI < 18.5:
            BMI_Category = 0
        elif BMI < 25:
            BMI_Category = 1
        elif BMI < 30:
            BMI_Category = 2
        else:
            BMI_Category = 3

        rbc = encode_col("rbc", request.form.get("rbc"))
        pc = encode_col("pc", request.form.get("pc"))
        pcc = encode_col("pcc", request.form.get("pcc"))
        ba = encode_col("ba", request.form.get("ba"))
        htn = encode_col("htn", request.form.get("htn"))
        dm = encode_col("dm", request.form.get("dm"))
        cad = encode_col("cad", request.form.get("cad"))
        appet = encode_col("appet", request.form.get("appet"))
        pe = encode_col("pe", request.form.get("pe"))
        ane = encode_col("ane", request.form.get("ane"))


        feature_vector = [
            age, bp, sg, al, su, bgr, bu, sc, sod, pot,
            hemo, pcv, wc, rc, BMI,
            rbc, pc, pcc, ba, htn, dm, cad, appet, pe, ane, BMI_Category
        ]

        X = np.array(feature_vector).reshape(1, -1)

        pred = model.predict(X)[0]

        if isinstance(pred, str):
            result = pred.upper()
        else:
            result = "CKD" if int(pred) == 1 else "No CKD"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
