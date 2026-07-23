import joblib
import gradio as gr

# ==========================================================
# Load Model
# ==========================================================
try:
    model = joblib.load("obesity_model.pkl")
except:
    model = joblib.load("obesity_model(1).pkl")


# ==========================================================
# Prediction Function
# ==========================================================
def predict_obesity(
    Age,
    Gender,
    Height,
    Weight,
    CALC,
    FAVC,
    FCVC,
    NCP,
    SCC,
    SMOKE,
    CH2O,
    family_history,
    FAF,
    TUE,
    MTRANS,
):

    input_data = [[
        Age,
        Gender,
        Height,
        Weight,
        CALC,
        FAVC,
        FCVC,
        NCP,
        SCC,
        SMOKE,
        CH2O,
        family_history,
        FAF,
        TUE,
        MTRANS
    ]]

    prediction = model.predict(input_data)[0]

    return f"🏥 Predicted Obesity Category:\n\n**{prediction}**"


# ==========================================================
# Theme
# ==========================================================
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="emerald",
    neutral_hue="slate",
)


# ==========================================================
# Interface
# ==========================================================
with gr.Blocks(theme=theme, title="Obesity Prediction Dashboard") as demo:

    gr.Markdown(
        """
# 🩺 Obesity Prediction Dashboard

### Predict Obesity Level using Machine Learning

---
### 👨‍💻 Developed by **Vansh**
### 🎓 Roll No. **241047**

Fill all details below and click **Predict**.
"""
    )

    with gr.Row():

        with gr.Column():

            Age = gr.Number(label="Age", value=25)

            Gender = gr.Dropdown(
                choices=["Male", "Female"],
                value="Male",
                label="Gender"
            )

            Height = gr.Number(
                label="Height (meters)",
                value=1.75
            )

            Weight = gr.Number(
                label="Weight (kg)",
                value=75
            )

            CALC = gr.Dropdown(
                choices=["no", "Sometimes", "Frequently", "Always"],
                value="Sometimes",
                label="Alcohol Consumption (CALC)"
            )

            FAVC = gr.Dropdown(
                choices=["yes", "no"],
                value="yes",
                label="Frequent High-Calorie Food (FAVC)"
            )

            FCVC = gr.Slider(
                1,
                3,
                value=2,
                step=1,
                label="Vegetable Consumption (FCVC)"
            )

            NCP = gr.Slider(
                1,
                4,
                value=3,
                step=1,
                label="Main Meals per Day (NCP)"
            )

        with gr.Column():

            SCC = gr.Dropdown(
                choices=["yes", "no"],
                value="no",
                label="Calories Monitoring (SCC)"
            )

            SMOKE = gr.Dropdown(
                choices=["yes", "no"],
                value="no",
                label="Smoking"
            )

            CH2O = gr.Slider(
                1,
                3,
                value=2,
                step=0.5,
                label="Daily Water Intake (CH2O)"
            )

            family_history = gr.Dropdown(
                choices=["yes", "no"],
                value="yes",
                label="Family History of Overweight"
            )

            FAF = gr.Slider(
                0,
                3,
                value=1,
                step=0.5,
                label="Physical Activity (FAF)"
            )

            TUE = gr.Slider(
                0,
                2,
                value=1,
                step=0.5,
                label="Technology Usage (TUE)"
            )

            MTRANS = gr.Dropdown(
                choices=[
                    "Automobile",
                    "Motorbike",
                    "Bike",
                    "Public_Transportation",
                    "Walking"
                ],
                value="Public_Transportation",
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
            Age,
            Gender,
            Height,
            Weight,
            CALC,
            FAVC,
            FCVC,
            NCP,
            SCC,
            SMOKE,
            CH2O,
            family_history,
            FAF,
            TUE,
            MTRANS
        ],
        outputs=output,
    )

    gr.Examples(
        examples=[
            [
                25,
                "Male",
                1.75,
                75,
                "Sometimes",
                "yes",
                2,
                3,
                "no",
                "no",
                2,
                "yes",
                1,
                1,
                "Public_Transportation"
            ],
            [
                40,
                "Female",
                1.60,
                95,
                "Frequently",
                "yes",
                1,
                4,
                "no",
                "no",
                1,
                "yes",
                0,
                2,
                "Automobile"
            ],
        ],
        inputs=[
            Age,
            Gender,
            Height,
            Weight,
            CALC,
            FAVC,
            FCVC,
            NCP,
            SCC,
            SMOKE,
            CH2O,
            family_history,
            FAF,
            TUE,
            MTRANS
        ],
    )

    gr.Markdown(
        """
---
### ⚠ Disclaimer
This prediction is generated by a Machine Learning model for educational purposes only and should not replace professional medical advice.
"""
    )


import os

port = int(os.environ.get("PORT", 7860))

demo.launch(
    server_name="0.0.0.0",
    server_port=port
)
