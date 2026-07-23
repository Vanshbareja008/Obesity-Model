import joblib
import numpy as np
import gradio as gr

# ==========================
# Load Model
# ==========================
model = joblib.load("obesity_model.pkl")


# ==========================
# Prediction Function
# ==========================
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
    try:
        input_data = np.array([[
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
            mtrans
        ]])

        prediction = model.predict(input_data)[0]

        if hasattr(model, "predict_proba"):
            confidence = max(model.predict_proba(input_data)[0]) * 100
            return f"Predicted Class: {prediction}\nConfidence: {confidence:.2f}%"

        return f"Predicted Class: {prediction}"

    except Exception as e:
        return f"Error: {e}"


# ==========================
# Interface
# ==========================
title = "# 🏥 Obesity Prediction System"

description = """
### Developed by **Vansh**
### Roll No. **241047**

Enter the patient's details and click **Predict**.
"""

demo = gr.Interface(
    fn=predict_obesity,
    inputs=[
        gr.Number(label="Age", value=22),

        gr.Dropdown(
            choices=["Male", "Female"],
            value="Male",
            label="Gender"
        ),

        gr.Number(label="Height (meters)", value=1.75),

        gr.Number(label="Weight (kg)", value=75),

        gr.Dropdown(
            choices=["no", "Sometimes", "Frequently", "Always"],
            value="Sometimes",
            label="CALC"
        ),

        gr.Dropdown(
            choices=["yes", "no"],
            value="yes",
            label="FAVC"
        ),

        gr.Number(label="FCVC", value=2),

        gr.Number(label="NCP", value=3),

        gr.Dropdown(
            choices=["yes", "no"],
            value="no",
            label="SCC"
        ),

        gr.Dropdown(
            choices=["yes", "no"],
            value="no",
            label="SMOKE"
        ),

        gr.Number(label="CH2O", value=2),

        gr.Dropdown(
            choices=["yes", "no"],
            value="yes",
            label="Family History with Overweight"
        ),

        gr.Number(label="FAF", value=1),

        gr.Number(label="TUE", value=1),

        gr.Dropdown(
            choices=[
                "Automobile",
                "Bike",
                "Motorbike",
                "Public_Transportation",
                "Walking"
            ],
            value="Public_Transportation",
            label="Transportation"
        ),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title=title,
    description=description,
    theme=gr.themes.Soft(),
    examples=[
        [
            21,
            "Male",
            1.75,
            70,
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
            "Public_Transportation",
        ]
    ],
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
