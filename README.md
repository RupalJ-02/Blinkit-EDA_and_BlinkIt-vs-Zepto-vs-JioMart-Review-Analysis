# Blinkit-EDA_and_BlinkIt-vs-Zepto-vs-JioMart-Review-Analysis
Analyzing Blinkit's sales performance, customer satisfaction and inventory distribution and A LSTM Sentiment Analysis Project on E-Commerce Reviews (Zepto vs Jiomart vs Blinkit).

# PART-A - Blinkit Data Analysis Dashboard

## Project Overview
This project presents a comprehensive Power BI dashboard for analyzing Blinkit's sales performance, customer satisfaction, and inventory distribution. The dashboard provides key insights and opportunities for optimization using various KPIs and visualizations.

## Key Performance Indicators (KPIs)
The dashboard focuses on the following primary KPIs:

* Total Sales: Overall revenue generated from all items sold ($1.20M)
* Average Sales: Average revenue per sale ($141)
* Number of Items: Total count of different items sold (8523)
* Average Rating: Average customer rating for items sold (3.9 out of 5)

# PART-B -LSTM Sentiment Analysis on Blinkit, Zepto, and Jiomart Reviews

This project performs sentiment analysis on e-commerce reviews using a Long Short-Term Memory (LSTM) model. The focus is on reviews from three specific platforms: Zepto, Jiomart, and Blinkit.

## Table of Contents

* [Introduction](#introduction)
* [Project Goal](#project-goal)
* [Dataset](#dataset)
* [Methodology](#methodology)
  * [Data Loading](#data-loading)
  * [Preprocessing](#preprocessing)
  * [Model Building](#model-building)
  * [Training and Evaluation](#training-and-evaluation)
  * [Prediction](#prediction)
* [Results](#results)
* [Power BI Dashboard](#PowerBIDashboard)
* [Conclusion](#conclusion)
* [Dependencies](#dependencies)
* [Usage](#usage)
* [Contributing](#contributing)
* [License](#license)

## Introduction

Sentiment analysis is a crucial task in natural language processing (NLP) that involves determining the emotional tone of a piece of text. This project applies sentiment analysis to e-commerce reviews, aiming to classify reviews as positive, negative, or neutral.

## Project Goal

The primary goal of this project is to demonstrate a practical application of deep learning, specifically LSTM models, for sentiment classification in the context of e-commerce reviews. The project uses reviews from Blinkit, Zepto, and Jiomart to train and evaluate the LSTM model.

## Dataset

The dataset for this project consists of e-commerce reviews collected from publicly available sources. The reviews are labeled with their corresponding sentiment (positive, negative, or neutral). The dataset is organized into separate files for Blinkit, Zepto, and Jiomart reviews.

## Methodology

The project follows a standard machine learning workflow:

### Data Loading

* The dataset is loaded into the Colab environment using the `pandas` library.
* The reviews and sentiment labels are extracted from the dataset.

### Preprocessing

* The text data is cleaned and preprocessed to remove noise, such as punctuation and special characters.
* The reviews are tokenized into individual words.
* Word embeddings are used to represent the words numerically.

### Model Building

* An LSTM model is constructed using the `keras` library.
* The model architecture includes an embedding layer, LSTM layers, and dense layers.

### LSTM (Long Short-Term Memory) Model: A Concise Explanation
* What they are:
A special type of recurrent neural network (RNN).
Designed to learn long-term dependencies in sequences (which regular RNNs struggle with).
* Key Feature: Memory Cells
Store information over time, like a computer's memory.
Controlled by three "gates":
Forget Gate: Decides what info to remove from memory.
Input Gate: Decides what new info to store in memory.
Output Gate: Decides what info to output based on memory.
* How they Work:
Process input sequences step-by-step.
At each step:
Cell updates its memory using the gates and current input.
Continues until the whole sequence is read.
Final output used for tasks like sentiment analysis.
* Why Good for Sentiment Analysis:
Sentiment depends on relationships between words across a text.
LSTMs are good at capturing these long-range connections.
In Essence:
LSTMs have a memory that selectively keeps and discards info.
This helps them "understand" the overall sentiment of text.

### Training and Evaluation

* The model is trained on the preprocessed dataset using an appropriate optimizer and loss function.
* The model's performance is evaluated using metrics such as accuracy, precision, and recall.

### Prediction

* The trained model is used to predict the sentiment of new, unseen reviews.
* The predictions are presented with their corresponding confidence scores.

## Results

The project's results show the effectiveness of the LSTM model in classifying e-commerce reviews with reasonable accuracy. The specific results, including the achieved accuracy, precision, and recall, are presented in the notebook.

Accuracy: 87.6%

Precision: 0.8

Recall: 0.9

F1 Score: 0.9

Prediction test example-

example_review = "I love the fast delivery and great service!"
print(example_review)
print(f"{predict_review(example_review, model, tokenizer, label_encoder, maxlen)}\n")

output-

I love the fast delivery and great service!
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 38ms/step
Sentiment: positive, Confidence: 0.97


## Power BI Dashboard

In addition to the sentiment analysis results presented in the notebook, this project also includes an interactive Power BI dashboard to visualize the key findings. The dashboard provides insights into the sentiment distribution, average rating, and other metrics across the three e-commerce platforms (Blinkit, Zepto, and Jiomart).

## Conclusion

This project demonstrates the viability of using deep learning techniques for sentiment analysis in the e-commerce domain. The LSTM model proves to be a valuable tool for understanding customer opinions and sentiment towards Blinkit, Zepto, and Jiomart.

## Dependencies

* Python 3.7+
* pandas
* NumPy
* scikit-learn
* TensorFlow
* Keras

## Usage

1. Clone the repository.
2. Upload the dataset files to your Google Colab environment.
3. Open the Colab notebook.
4. Execute the code cells in the notebook sequentially.

## Contributing

Contributions to this project are welcome. You can contribute by:

* Improving the model's performance.
* Adding new features.
* Fixing bugs.
* Providing feedback.

To contribute, please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.
