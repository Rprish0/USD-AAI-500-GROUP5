# Amazon Fine Food Reviews Sentiment Analysis

This project analyzes customer reviews from Amazon's fine food products to determine sentiment as positive, negative, or neutral. Leveraging Natural Language Processing (NLP) techniques and various machine learning algorithms, the project provides insights into customer satisfaction and product performance.

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Project Goals](#project-goals)
4. [Technologies Used](#technologies-used)
5. [Model Implementation](#model-implementation)
6. [Results](#results)
7. [Future Work](#future-work)
8. [Authors](#authors)

---

## Introduction

Sentiment analysis is a subfield of NLP that aims to classify opinions expressed in text as positive, neutral, or negative. This project applies sentiment analysis to Amazon fine food reviews, exploring traditional and modern machine learning models to classify sentiments effectively.

## Dataset

- **Source**: [Amazon Fine Food Reviews](https://www.kaggle.com/snap/amazon-fine-food-reviews)
- **Size**: 568,454 reviews
- **Features**:
  - `Id`, `ProductId`, `UserId`, `ProfileName`
  - `HelpfulnessNumerator`, `HelpfulnessDenominator`
  - `Score` (ratings from 1 to 5)
  - `Summary`, `Text`
- **Objective**: Classify reviews with ratings of 4-5 as positive, 1-2 as negative, and ignore neutral (rating 3).

## Project Goals

1. Preprocess and clean raw text data for better sentiment classification.
2. Implement and evaluate different machine learning algorithms.
3. Compare the performance of Logistic Regression, SVM, and Deep Learning models.

## Technologies Used

- **Programming Language**: Python
- **Libraries**:
  - NLP: `nltk`, `scikit-learn`
  - Machine Learning: `Keras`, `gensim`
  - Data Manipulation: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`

## Model Implementation

1. **Data Preprocessing**:
   - Removed punctuation, special characters, and stopwords.
   - Applied lemmatization and TF-IDF vectorization.
   - Addressed class imbalance through oversampling.

2. **Algorithms Used**:
   - Logistic Regression
   - Support Vector Machines (SVM)
   - Deep Learning (feedforward neural networks with dropout regularization)

3. **Evaluation Metrics**:
   - Accuracy
   - Precision, Recall, F1-Score

## Results

| Model                | Accuracy | Positive (F1) | Neutral (F1) | Negative (F1) |
|----------------------|----------|---------------|--------------|---------------|
| Logistic Regression | 88.5%    | 88.5%         | 85.5%        | 87.5%         |
| SVM                 | 88.3%    | 88%           | 84.5%        | 86.5%         |
| Deep Learning       | 89.2%    | 89.5%         | 85.5%        | 87.5%         |

- **Key Observations**:
  - Logistic Regression is efficient and interpretable, making it a robust baseline.
  - SVM performs comparably to Logistic Regression but is computationally expensive.
  - Deep Learning offers the best accuracy but requires significant resources.

## Future Work

1. Incorporate advanced embeddings like Word2Vec or BERT.
2. Fine-tune the deep learning architecture for improved performance.
3. Use larger and more diverse datasets to enhance generalization.

## Authors

- **Rishabh Pathak**
- **Shubham Gondralwar**
- **Narendra Iyer**

---

GitHub Repository: [USD-AAI-500-GROUP5](https://github.com/Rprish0/USD-AAI-500-GROUP5)
