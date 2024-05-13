<h1>Text Classification to Detect Sarcasm</h1>

 ### [YouTube Demonstration](https://youtu.be/7eJexJVCqJo)

<h2>Description</h2>

In this project I will build regression and classification models, with and emphasis on how the complexity of a model influences its generalization performance. My central thes is to explore the common scenario in machine learning where data is presumed to come from an unobservable 'gold standard' model. By using regression techniques I will aim to approximate the model to achieve accurate predictions on future data. The initial underlying model is a polynomial of unknown degree. As the project progresses I refine these assumptions, employing methods like Lasso to incorporate sparsity in the model's coefficients. The tasks are interlinked, with the outputs of earlier tasks feeding into later ones. 


<br />


<h2>Python Libraries Used</h2>

<b>NumPy (numpy): For numerical computing and operations on arrays.</b>
<b>Pandas (pandas): For data manipulation and analysis.</b>
<b>Scikit-learn (sklearn.model_selection): For splitting datasets into training and testing sets.</b>
<b>Matplotlib (matplotlib.pyplot): For plotting and data visualization.</b>



<h2>Environments Used </h2>

- <b>Jupyter</b> (21H2)



<h2>Program walk-through:</h2>

<p align="left">









<br />
Load libraries
<img src="https://i.imgur.com/TXaS04D.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/><br />







Raw Data with Noise Generation
I will first run the following cell to generate and plot the data points used throughout the project.

The independent variable  𝑥  consists of  𝑛 evenly spaced points from the interval  [0,20]
The dependent variable  𝑦=0.05𝑥3−𝑥2−𝑥+𝐶𝜖 is a function of  𝑥 where  𝜖∼(0,1) represents the standard Gaussian noise and  𝐶   is a constant indicating the noise magnitude.<br />

<img src="https://i.imgur.com/eSZX0J6.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>





<br />
<br />
Raw Data Plot :  <br/>
<img src="https://i.imgur.com/On1m4Sh.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>

Plot <br />
<img src="https://i.imgur.com/POKmrVP.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>




<br />
<br />
Polynomial Features:  <br/>
I will use polynomial regression to explore the relationship between two variables, \( x \) and \( y \). This is because a simple linear function isn't complex enough to accurately describe their relationship. Instead, I will consider \( y \) as a polynomial function of \( x \), which includes \( x \) raised to various powers (like \( x^2, x^3, \) etc.), each multiplied by different coefficients. 

Although \( y \) is a non-linear function of \( x \), it is linear in terms of these powers of \( x \). I can still use linear regression techniques by transforming \( x \) into its higher powers. To create these transformed features, I will use the PolynomialFeatures tool from the scikit-learn library. This approach allows me to model more complex relationships between \( x \) and \( y \) using a technique that still fundamentally relies on linear regression principles.

Steps:

1. Fit Polynomial Models: Write code to fit polynomial expansions of the training data `X_train` for degrees 1, 3, 7, and 11 using a `LinearRegression` model. I will use `PolynomialFeatures` to transform the original `X_train` data by adding new polynomial feature columns for the specified degrees.

2. Generate Predictions: For each of the polynomial models fitted in step 1, I will generate predictions using 100 evenly spaced points from 0 to 20. I will use `np.linspace(0, 20, 100).reshape(-1, 1)` to create the input matrix of x-values. I will then transform these x-values using `fit_transform` to include the polynomial features before feeding them into the `predict` method of the trained linear regression model. After obtaining predictions, I will transform the output into a single row with 100 columns using `transpose()`.

3. Assemble Results: Combine the predictions from all polynomial models into a single numpy array. This array will have 4 rows (one for each polynomial degree: 1, 3, 7, 11) and 100 columns, representing the predictions at each of the 100 points for each model.

4. Output the Array: The function will return this numpy array, which has the shape (4, 100). This array will be used for further analysis and validation. 

These steps are to explore how polynomial features of different degrees fit the training data and affect predictions across a specified interval.
<br />




Additional Libraries:
<img src="https://i.imgur.com/HmHgoEX.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>



<br />
<br />
Function:  <br/>
<img src="https://i.imgur.com/fwyb7BX.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>





<br />
<br />
Array:  <br/>
<img src="https://i.imgur.com/4jT9VzG.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>




<br />
<br />
Polynomial Curves plot #1:  <br/>
I will visualize the polynomials learned from the training data, along with the training and the testing data.
<br />



<img src="https://i.imgur.com/z4yhEJO.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/><br />


<img src="https://i.imgur.com/xIWEEIi.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/><br />



<br />
<br />
Polynomial Features and Quality of Fit :  <br/>

I will create a function that constructs polynomial `LinearRegression` models using the training data `X_train` for the polynomial degrees 1, 3, 7, and 11. After fitting each model, I will calculate the R² (coefficient of determination) score, which assesses the quality of the fit, for both the training and testing data sets.

This function will return a tuple containing two lists: `r2_train` and `r2_test`. The `r2_train` list will include the R² scores for the training data, while the `r2_test` list will contain the scores for the testing data. I will utilize the `r2_score` function from `sklearn.metrics` to compute these scores.



<img src="https://i.imgur.com/IhjRfyl.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>


<br />
<br />
Solution:  <br/>
<img src="https://i.imgur.com/VbBIZFD.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>


<br />
<br />
KNN Regression:  <br/>
Fit a KNN regression model with the training data and return the  𝑅2   value on the testing data. Use the default hyper-parameters.

This function will return a single float value.


<img src="https://i.imgur.com/3KaS4AL.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/><br />
<img src="https://i.imgur.com/DH0jqxy.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/><br />

<img src="https://i.imgur.com/L46snKj.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/><br />




<br />
<br />
Polynomial Model Degree fit: under, over, optimal :  <br/>

Based on the  𝑅2  scores from Task 1b, which degree of the polynomial causes the model to be

1. underfitting;
2. overfitting; or
3. achieving good generalisation performance?
I will plot the degrees of the polynomial against the  𝑅2  scores to visualise their relationship.

The function will return a 3-tuple with the degree values in this order: (Underfitting, Overfitting, Good_Generalization) 


<img src="https://i.postimg.cc/Pf0FH7jn/18.jpg" height="80%" width="80%" alt="Disk Sanitization Steps"/>




<br />
<br />
Polynomial Fitting With Lasso Regression:  <br/>

Training models on high-degree polynomial features can result in overly complex models that overfit the training data.
I considered adding some regularization to constrain the model complexity.<br />

Steps:
1. Compare the non-regularized LinearRegression model (with the default hyper-parameters), to a new regularised Lasso Regression model (with hyper-parameters alpha=0.01, max_iter=10000) --- on polynomial features of varying degrees.
2. Observe the difference to see the difference with the polynomials that were fit in task 1.

Function returns predictions for the regularized model. 
     Generate predictions for 100 evenly spaced points on the interval [0, 20] 
     Store the results in a numpy array, whose the first row stores the predictions from the model of degree 1, the second row stores the predictions       from the model of degree 3 and so on.

Function returns a numpy array of the shape (4, 100).


<img src="https://i.postimg.cc/mD4KnB3B/19.jpg" height="80%" width="80%" alt="Disk Sanitization Steps"/>



<br />
<br />
<img src="https://i.imgur.com/AeZkvFQ.png" height="80%" width="80%" alt="Disk Sanitization Steps"/>



<br />
<br />
NEXT PICTURE:  <br/>
<img src="https://i.imgur.com/AeZkvFQ.png" height="80%" width="80%" alt="Disk Sanitization Steps"/>



<br />
<br />
<br/>
<img src="https://i.postimg.cc/Xq0mT3wn/21.jpg" height="80%" width="80%" alt="Disk Sanitization Steps"/>








<br />
<br />
Lasso Regression vs a 'gold standard' dataset :  <br/>
Return the  𝑅2 score for each of the Lasso models above relative to a new 'gold standard' test set generated from the true underlying cubic polynomial model without noise. Test is run by computing the true noise-less underlying function t^3/20 - t^2 - t for each of 100 evenly spaced points on the interval [0, 20] . 

For each degree (1, 3, 7, 11), the 𝑅2 score is computed using this 'gold standard' test set and returned the polynomial degree that gives the best fit on the 'gold standard' test set. 

Function returns an integer, in the set (1,3,7,11). 

Question
Does the optimal polynomial degree match the true polynomial degree?



<img src="https://i.postimg.cc/wjTZ8gGk/22.jpg" height="80%" width="80%" alt="Disk Sanitization Steps"/><br />


<img src="https://i.postimg.cc/hGwV41PG/23.jpg" height="80%" width="80%" alt="Disk Sanitization Steps"/><br />









</p>

<!--
 ```diff
- text in red
+ text in green
! text in orange
# text in gray
@@ text in purple (and bold)@@
```
--!>
