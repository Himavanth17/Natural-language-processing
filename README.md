### Import Libraries

```python

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

```

- pandas: A tool to handle and analyze data easily.
- matplotlib.pyplot: A tool to make pictures (graphs).
- numpy: A tool to handle numbers and math stuff.

### 2. Read the Data

```python

df = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

```

- Read data from a file called 'Restaurant_Reviews.tsv'. The file has reviews of restaurants.
- delimiter='\t' means the file uses tabs to separate columns.
- quoting=3 means ignoring quotation marks.

### 3. Text Cleaning and Preparation

```python

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

```

- re: A tool to work with text (like finding words).
- nltk: A tool for working with human language.
- nltk.download('stopwords'): Download common words like 'the', 'is', 'and'.
- stopwords: These are the common words.
- PorterStemmer: A tool to reduce words to their base form (like 'running' to 'run').

### 4. Clean Each Review

```python

corpus = []
for i in range(0, 1000):
  review = re.sub('[^a-zA-Z]', ' ', df['Review'][i])
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)

```

- corpus = []: Create an empty list to store cleaned reviews.
- for i in range(0, 1000): Loop through the first 1000 reviews.
- re.sub('[^a-zA-Z]', ' ', df['Review'][i]): Replace anything that's not a letter with a space.
- review.lower(): Change all letters to lowercase.
- review.split(): Split the review into individual words.
- ps = PorterStemmer(): Create a tool to get the base form of words.
- all_stopwords = stopwords.words('english'): Get a list of common words.
- all_stopwords.remove('not'): Keep the word 'not' because it's important.
- review = [ps.stem(word) for word in review if not word in set(all_stopwords)]: Keep only important words in their base form.
- review = ' '.join(review): Join the words back into a single string.
- corpus.append(review): Add the cleaned review to the list.

### 5. Create a Bag of Words Model

```python

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
x = cv.fit_transform(corpus).toarray()
y = df.iloc[:, -1].values

```

- CountVectorizer: A tool to turn text into numbers.
- cv = CountVectorizer(max_features=1500): Create a model to keep the 1500 most important words.
- x = cv.fit_transform(corpus).toarray(): Turn the cleaned reviews into a big table of numbers.
- y = df.iloc[:, -1].values: Get the last column of the data (whether the review is good or bad).

### 6. Split Data into Training and Testing Sets

```python

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

```

- train_test_split: A tool to split data into two parts.
- x_train, x_test, y_train, y_test: The training and testing data.
- test_size=0.25: Use 25% of the data for testing.
- random_state=0: Keep the split the same every time.

### 7. Train the Model

```python

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

```

- GaussianNB: A tool to learn from the data.
- classifier = GaussianNB(): Create a learning tool.
- classifier.fit(x_train, y_train): Teach the tool using the training data.

### 8. Make Predictions

```python

from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(x_test)

```

- confusion_matrix, accuracy_score: Tools to measure how good the model is.
- y_pred = classifier.predict(x_test): Predict the results for the test data.

### 9. Evaluate the Model

```python

ac = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

```

- ac = accuracy_score(y_test, y_pred): Calculate how many predictions are correct.
- cm = confusion_matrix(y_test, y_pred): Create a table showing the results of predictions vs actual results.

### Summary

This code reads restaurant reviews, cleans and processes them, and then uses a machine learning model to predict whether new reviews are good or bad. It also evaluates how well the model performs.
