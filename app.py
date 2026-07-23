import joblib
import pandas as pd
import gradio as gr

# ==========================================================
# Load Model
# ==========================================================
try:
    model = joblib.load("obesity_model.pkl")
    model_loaded = True
except Exception as e:
    print(e)
    model_loaded = False


# ==========================================================
# Prediction Function
# ==========================================================
def predict_obesity(
    age,
    gender,
    height,
    weight,
    calc,
    favc,
    fcvc,
    ncp,
    scc,
    smoke,
    ch2o,
    family_history,
    faf,
    tue,
    mtrans,
):

    if not model_loaded:
        return "❌ Model could not be loaded."

    data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Height": height,
        "Weight": weight,
        "CALC": calc,
        "FAVC": favc,
        "FCVC": fcvc,
        "NCP": ncp,
        "SCC": scc,
        "SMOKE": smoke,
        "CH2O": ch2o,
        "family_history_with_overweight": family_history,
        "FAF": faf,
        "TUE": tue,
        "MTRANS": mtrans
    }])

    prediction = model.predict(data)[0]

    if hasattr(model, "predict_proba"):
        confidence = max(model.predict_proba(data)[0]) * 100
        return f"""
### Prediction

**Obesity Level:** {prediction}

**Confidence:** {confidence:.2f}%
"""
    else:
        return f"### Prediction\n\n**Obesity Level:** {prediction}"


# ==========================================================
# Dashboard
# ==========================================================
with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="emerald"
    ),
    title="Obesity Prediction Dashboard"
) as demo:

    gr.Markdown("""
# 🏥 Obesity Prediction Dashboard

### Machine Learning Based Health Assessment

Predict obesity category using lifestyle and physical parameters.

---
**Developed by:** **Vansh**  
**Roll No.:** **241047**
---
""")

    with gr.Row():

        with gr.Column():

            age = gr.Number(label="Age", value=25)

            gender = gr.Dropdown(
                ["Male", "Female"],
                value="Male",
                label="Gender"
            )

            height = gr.Number(label="Height (meters)", value=1.75)

            weight = gr.Number(label="Weight (kg)", value=72)

            calc = gr.Dropdown(
                ["no", "Sometimes", "Frequently", "Always"],
                value="Sometimes",
                label="Alcohol Consumption (CALC)"
            )

            favc = gr.Dropdown(
                ["yes", "no"],
                value="yes",
                label="High Calorie Food (FAVC)"
            )

            fcvc = gr.Slider(
                1,
                3,
                value=2,
                step=1,
                label="Vegetable Consumption (FCVC)"
            )

            ncp = gr.Slider(
                1,
                4,
                value=3,
                step=1,
                label="Meals per Day (NCP)"
            )

        with gr.Column():

            scc = gr.Dropdown(
                ["yes", "no"],
                value="no",
                label="Calories Monitoring (SCC)"
            )

            smoke = gr.Dropdown(
                ["yes", "no"],
                value="no",
                label="Smoking"
            )

            ch2o = gr.Slider(
                1,
                3,
                value=2,
                step=0.5,
                label="Water Intake (CH2O)"
            )

            family = gr.Dropdown(
                ["yes", "no"],
                value="yes",
                label="Family History"
            )

            faf = gr.Slider(
                0,
                3,
                value=1,
                step=0.5,
                label="Physical Activity (FAF)"
            )

            tue = gr.Slider(
                0,
                2,
                value=1,
                step=0.5,
                label="Technology Usage (TUE)"
            )

            mtrans = gr.Dropdown(
                [
                    "Automobile",
                    "Bike",
                    "Motorbike",
                    "Public_Transportation",
                    "Walking"
                ],
                value="Walking",
                label="Transportation"
            )

    predict_btn = gr.Button(
        "🔍 Predict Obesity Level",
        variant="primary"
    )

    output = gr.Markdown()

    predict_btn.click(
        predict_obesity,
        inputs=[
            age,
            gender,
            height,
            weight,
            calc,
            favc,
            fcvc,
            ncp,
            scc,
            smoke,
            ch2o,
            family,
            faf,
            tue,
            mtrans
        ],
        outputs=output
    )

    gr.Examples(
        examples=[
            [25,"Male",1.75,72,"Sometimes","yes",2,3,"no","no",2,"yes",1,1,"Walking"],
            [35,"Female",1.60,95,"Frequently","yes",1,4,"no","no",1.5,"yes",0,2,"Automobile"],
            [19,"Male",1.82,68,"no","no",3,3,"yes","no",3,"no",3,0.5,"Bike"]
        ],
        inputs=[
            age,
            gender,
            height,
            weight,
            calc,
            favc,
            fcvc,
            ncp,
            scc,
            smoke,
            ch2o,
            family,
            faf,
            tue,
            mtrans
        ]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
