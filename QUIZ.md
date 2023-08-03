*** Note: cần tìm hiểu rõ hơn về phần Bias và Variance, ML development process (C2_W2, C2_W3).

# C1_W1
## Practice quiz: Regression
1. For linear regression, the model is f(w, b)) = wx + b. Which of the following are the inputs, or features, that are fed into the model and with which the model is expected to make a prediction?
    <br>***x*** 
    <br>m
    <br>(x, y)
    <br>w and b
2. For linear regression, if you find parameters w and b so that J(w, b) is very so close to zero, what can you conclue?
    <br> **The selected values of the parameters w and b cause the algorithm to fit the training set really well.**
    <br> The selected values of the parameters w and b cause the algorithm to fit the training set really poorly.
    <br> This is never possible - there must be a bug in the code

## Practice quiz: Supervised vs Unsupervised Learning
1. Which are the two common types of supervised learning?
    <br> **Regression**
    <br> Clustering
    <br> **Classification**
2. Which of these is a typo of unsupervised learning?
    <br> **Clustering**
    <br> Classification
    <br> Regression

## Practice quiz: Train the model with gradient descent
1. Gradient descent is an algorithm for find values of parameters w and b that minimize the cost function J. When ∂J/∂w is a negative number, what happens to w after one update step?
    <br> **w increases**
    <br> w decreases
    <br> w stays the same
    <br> It is not possible to tell if w will increase or decrease
2. For linear regression, what is the update step for parameter b?
    <br> b = b - learning_rate * (1/m) * Σ(f_wb(xi) - yi)xi với i = 0 -> m - 1
    **<br> b = b - learning_rate * (1/m) * Σ(f_wb(xi) - yi) với i = 0 -> m - 1**

# C1_W2
## Practice quiz: Gradient descent in practice
1. Of the circumstances below, which is the most important to use feature scaling?
    **<br> Feature scaling is helpful when one feature is much larger (or smaller) than the another feature.**
    <br> Feature scaling is helpful when all features in the origin data (before scaling is applied) range from 0 to 1
2. You are helping a grocery store predict its revenue, and have data on its items sold per week, and price per item. What could be a useful engineered features?
   <br> For each product, calculate the number of items sold times price item
   **<br> For each product, calculate the number of items sold divided by the price per item**

## Multiple linear regression
![img_5.png](assets/img_5.png)
![img_6.png](assets/img_6.png)

# C1_W3

## Practice quiz: Cost function for logistic regression
![img_15.png](assets/img_15.png)
## Practice quiz: Gradient descent for logistic regression
![img_16.png](assets/img_16.png)

# C2_W1
![img_18.png](assets/img_18.png)
1d sai (a^-1 -> thành a^0)
![img_19.png](assets/img_19.png)
![img_20.png](assets/img_20.png)
![img_21.png](assets/img_21.png)
![img_22.png](assets/img_22.png)

![img_25.png](assets/img_25.png)
![img_26.png](assets/img_26.png)

# C2_W2
![img_27.png](assets/img_27.png)
![img_29.png](assets/img_29.png)
![img_30.png](assets/img_30.png)

# C2_W3
![img_33.png](assets/img_33.png)
![img_34.png](assets/img_34.png)
![img_36.png](assets/img_36.png)
(xem lại bên readme)
![img_37.png](assets/img_37.png)

# C2_W4
![img_38.png](assets/img_38.png)

![img_39.png](assets/img_39.png)
![img_40.png](assets/img_40.png)
![img_41.png](assets/img_41.png)
![img_42.png](assets/img_42.png)

# C3_W1
![img_43.png](assets/img_43.png)
![img_44.png](assets/img_44.png)
# C3_W2

# C3_W3

[//]: # (![img_45.png]&#40;img_45.png&#41;)

[//]: # (![img_46.png]&#40;img_46.png&#41;)

[//]: # ()
[//]: # (![img_47.png]&#40;img_47.png&#41;)

[//]: # ()
[//]: # (![img_48.png]&#40;img_48.png&#41;)