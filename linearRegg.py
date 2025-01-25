import numpy as np
from sklearn import linear_model, datasets
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from fpdf import FPDF
import streamlit as st
from io import BytesIO

# Load dataset
diabetes = datasets.load_diabetes()
diabetes_x = diabetes.data[:, np.newaxis, 2]
diabetes_x_train = diabetes_x
diabetes_x_test = diabetes_x
diabetes_y_train = diabetes.target
diabetes_y_test = diabetes.target

# Train model
model = linear_model.LinearRegression()
model.fit(diabetes_x_train, diabetes_y_train)
diabetes_y_predicted = model.predict(diabetes_x_test)

# Function to get model statistics
def get_receipt():
    mse = mean_squared_error(diabetes_y_test, diabetes_y_predicted)
    weights = model.coef_
    intercept = model.intercept_
    receipt_text = f"Mean Squared Error: {mse:.2f}\nWeights: {weights[0]:.2f}\nIntercept: {intercept:.2f}"
    return receipt_text

# Function to create a plot
def create_plot():
    plt.figure(figsize=(8, 6))
    plt.scatter(diabetes_x_test, diabetes_y_test, color="blue", label="Actual Data")
    plt.plot(diabetes_x_test, diabetes_y_predicted, color="red", label="Predicted Line")
    plt.title("Diabetes Linear Regression")
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.legend()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return buf

# Function to create a PDF report
def create_pdf_with_image(plot_buffer, receipt_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(0, 10, "Report of my ML Model", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, receipt_text)
    pdf.ln(10)
    pdf.image(plot_buffer, x=10, y=None, w=180)
    pdf_output = BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)
    return pdf_output

# Streamlit UI
st.title("Diabetes Linear Regression Report Generator")

# Generate model statistics
receipt = get_receipt()
st.write("### Model Statistics")
st.text(receipt)

# Generate and display the plot
st.write("### Plot of Predicted vs Actual")
plot_buffer = create_plot()
st.image(plot_buffer, caption="Regression Plot", use_column_width=True)

# Generate PDF and provide download option
if st.button("Generate PDF Report"):
    pdf_buffer = create_pdf_with_image(plot_buffer, receipt)
    st.download_button(
        label="Download PDF Report",
        data=pdf_buffer,
        file_name="diabetes_report.pdf",
        mime="application/pdf",
    )


