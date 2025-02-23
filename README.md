# Salizwa-Ntlabati01
PGDIP Assignment 2: Create a Simple AI Model for Linear Regression Using Rust Burn Library VERSION 0.16

# 🎓 Student Pass Rate Prediction using Linear Regression in Rust

This project implements a **simple linear regression model** using the **Rust Burn library** to predict **students' pass rates** based on the number of **study hours**.  The goal is to establish a linear relationship between study hours and the likelihood of passing a subject.

## 📖 Overview
The model follows the equation:

\[
\text{Pass Rate} = \text{Weight} \times \text{Study Hours} + \text{Bias}
\]

It learns to predict how likely students are to pass based on how much they study.

---

## 🛠 Approach

1. Dataset: The dataset consists of study hours (X) and pass rate (%) (Y):
2. Linear Regression Formular applied:
   - **Y = W.X + B ( Pass Rate = Weight x Study Hours + Bias)**
       - Y= Predicted pass rate (Outcome)
       - X = Study hours (Input)
       - W (Weight) = Effect of studying on pass rate (slop)
       - B (Bias) = Base pass rate with 0 study hours (Intercept)
     - Epochs = Number of times the model updates W and B to minimize errors
    
  ## 👟 The Model Training Process

  - ✅ starts with the random values  weight and bias.
  - ✅ Uses gradient descent to adjust values based on Mean Squared Error (MSE):

    ![image](https://github.com/user-attachments/assets/46641ff8-be20-4bbc-8888-3548113fa119)
  
   - ✅ Update weights and bias using Gradient Descent:

      ![image](https://github.com/user-attachments/assets/35fd21e2-4dcb-4488-b993-f978bdb662dc)

  - ✅ Runs for 500 epochs, improving accuracy.


## 📊 Results & Model Evaluation

After training, the model prints the final weight and bias values, and predicts the pass rate for different study hours.

**Example Output:**

Epoch 0: Loss = 502.3, Weight = 0.4567, Bias = 0.2345
Epoch 50: Loss = 42.3, Weight = 5.4321, Bias = 40.7892

Epoch 500: Loss = 1.2, Weight = 5.7527, Bias = 39.7596
Trained Model: Weight = 5.75, Bias = 39.76


**Predicted Pass Rate**

![<img width="416" alt="image" src="https://github.com/user-attachments/assets/6c4b90e2-dc69-4533-9cf7-aa1d04bc235c" />]


## 🤔 Reflection ##

**Lessons Learned:**

- Linear regression is effective for modeling relationships between numerical variables.

- Gradient descent helps optimize the model by minimizing errors.

- Training for multiple epochs improves accuracy, but too many can cause overfitting.

**Future Improvements:**

- Use a larger dataset from an external source (e.g., CSV, database).

- Implement multiple regression by adding more features (e.g., attendance, test scores).


## 📜 Resources Used ##

- Youtube: https://www.youtube.com/watch?v=dn8kjbU2J4U
- JetBrains
- Udemy


     
