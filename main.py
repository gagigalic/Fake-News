import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report

data = pd.read_csv("news.csv")

#shape
print(data.shape)

#info
print(data.info())

#describe
print(data.describe())

print(data.head())

labels = data.label
print(labels.head())

value_counts = data['label'].value_counts()
print(value_counts)

plt.figure(figsize=(8, 6))
ax = sns.countplot(x='label', data=data)
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')
plt.title('Number of fake and true values')
plt.xlabel('Label')
plt.ylabel('Number of instances')
plt.savefig("plot.png")
plt.close()


#training and testing sets
x_train,x_test,y_train,y_test=train_test_split(data['text'], labels, test_size=0.2, random_state=7)

tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

tfidf_train=tfidf_vectorizer.fit_transform(x_train)
tfidf_test=tfidf_vectorizer.transform(x_test)

pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

predictions=pac.predict(tfidf_test)
score=accuracy_score(y_test,predictions)
print(f'Accuracy: {round(score*100,2)}%')

print(classification_report(predictions,y_test))
print(confusion_matrix(predictions,y_test))

#example
example = "In a groundbreaking discovery, scientists have found a new species of flying penguins in Antarctica. These penguins are said to have evolved the ability to glide through the air, covering distances of up to 100 meters"
new_data = pd.DataFrame({'text': [example]})

new_data_tfidf = tfidf_vectorizer.transform(new_data['text'])
new_predictions = pac.predict(new_data_tfidf)

print(new_predictions)


