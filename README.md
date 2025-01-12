# cs412-project
## Classification Task
In the provided dataset, we found the features most relevant to our task to be the category names and post captions. We also considered using biographies, however we couldn’t observe an increase in our accuracy when we did so. We converted the post captions and the category names into a TF-IDF matrix via the provided code, in order to use the textual data for our classification task.

Before ending up with our final model, we experimented with several other approaches. At first, we manually mapped each Instagram category name of the accounts in the training data to the closest of the 10 categories in our task and created a model that classifies accounts based on the Instagram category names, and uses the naïve bayes classifier for accounts without category names. This model did not have a significantly higher accuracy than the baseline model. Possible reasons for this are many accounts not having category names and the category name feature not being as decisive of a feature as once thought. We also experimented with models that perform contextual analysis on the text data and map the account to the closest category among the 10 categories, however we observed that the models we used made mistakes even on very straightforward data, and the classification accuracy was not quite high as well.

Finally, we decided to use an ensemble voting classifier model with soft voting, with the following algorithms:
 - SVM
- Gradient Boosting
- Naïve Bayes
- Random Forest

SVM and Gradient Boosting are mainly responsible for the high validation accuracy that we obtained. We found SVM to be the most suitable algorithm for our task. SVM is excellent when dealing with a text classification problem where the TF-IDF matrix at hand is sparse and high-dimensional, which is also the case for our problem. Gradient Boosting does not perform as good as SVM in sparse datasets, however it is still a very robust algorithm for a classification tasks, so we added it to our ensemble. Naïve Bayes is not exactly a top performer for our task, however it is surprisingly effective for such a simple model and it did slightly increase our validation accuracy when we incorporated it to our ensemble, so we decided to keep it. Same for Random Forest.
|  Models | Algorithms in the ensemble | Data Used | Validation Accuracy
|--|--| --|--|
| Baseline | Naive Bayes |Post Captions| 57%|
| Initial Model | SVM, Naïve Bayes, Random Forest |Post Captions| 65%|
| Final Model| SVM, Naïve Bayes, Random Forest, Gradient Boosting |Post Captions and Instagram Category Names| 69%|

At first, we used an ensemble model containing SVM, Naïve Bayes, and Random Forest. Our validation accuracy was 65%, which was a good increase compared to the 57% accuracy in the baseline. We then incorporated category names to our TF-IDF matrix and added Gradient Boosting to the ensemble, and obtained a validation accuracy of 69%. This data is organized in Figure 1. (We re-trained the model several times and the validation accuracy sometimes had severe fluctuations. The accuracy shown in the Python file at GitHub might not be exactly on par with the data shared here.)

We opted to use soft voting instead of hard voting in our ensemble, as we found that probabilities are useful for voting in our ensemble model.

Lastly, while exploring the data, we realized that there are many overlappings between the instances in the training data and the test data, which meant that we had access to the ground truth for a portion of the predictions that we have to make. We added a couple lines of code to our model so that after the model is done with the predictions, the predictions for the overlapping instances are replaced with the ground truth.
## Regression Task


#### Overview of the repository for regression
The repository develops and evaluates regression models to predict Instagram post-like counts. The data preprocessing scripts load and parse the input datasets, such as the training dataset stored in a compressed JSON file. These scripts also process features, including timestamps, user profiles, and post characteristics, to create a structured dataset suitable for modeling.

 - **Timestamp** converts the string timestamp to an integer value to compare them.

 - **average like count** is used as a feature but we tried to develop it by taking time consideration into account. Then, *predict_like_count_time_weigheted* is created as seen in Figure 1.

 - **predict_like_count_time_weigheted**  takes 5 nearest like count numbers to take the average.  This does not change the result too much so it is not added to the main algorithm.


The main algorithm uses the **gradient boost** algorithm to predict the like count of the posts. The features in this algorithm are user's

 - **comments count**
 - **timestamp**
 - **average like count**

Average like count and comments count are transformed using the log function to account for high values seen in the graph. When the final prediction is taken, the inverse log is used to extract real like count. 

#### Methodology for regression

Firstly, various machine learning algorithms, including Random Forest and K-Nearest Neighbors (KNN), were implemented to predict the like counts of Instagram posts. Both models showed potential in handling the structured data and capturing relationships between features such as follower count, post activity, and timestamps. However, upon evaluation, the gradient-boosting algorithm demonstrated superior performance in terms of accuracy and efficiency. 

Secondly, the parameters used in the GradientBoostingRegressor model are  chosen to generalize data. We determined model parameters such as number of estimators and maximum tree depth by trial on first 30 users' posts.

#### Results

Overall, the model appears to perform reasonably well in predicting the like count for posts, although there are some notable discrepancies between predicted and actual values exist. While many predictions are close to the actual like counts, there are some significant errors where the model overestimates or underestimates the number of likes. These outliers indicate areas for improvement, especially in predicting extreme values. However, in general, the model seems to be capturing trends and providing useful predictions, though further adjustments could help fine-tune its accuracy.

Looking at Graph 1 how well the model predicts like counts on social media posts, we can see some interesting patterns. The graph shows that most posts get relatively few likes, and the model does a good job of predicting this common pattern. When we look at the orange bars (predictions) compared to the blue bars (actual likes), they match quite well, especially for posts with lower like counts. The model seems to understand that viral posts are rare but possible. We can see this in the few cases where posts got between 45,000 to 60,000 likes. While the model didn't get these numbers exactly right, it did predict that some posts would reach these high numbers. What's most encouraging is how well the predictions work for everyday posts. Most content falls in the range of 0-10,000 likes, and here the model's predictions closely follow the real numbers. This means the model is most reliable for predicting typical engagement levels, which is what most users would care about. However, it's worth noting that predicting exact numbers for viral posts (those with very high like counts) is still challenging. This makes sense because viral success can be unpredictable and influenced by many factors that might be hard for the model to account for. Overall, the model shows promising accuracy for everyday predictions while maintaining a reasonable understanding of the possibility of viral success. This makes it a useful tool for understanding typical social media engagement patterns.

 The model shows strong accuracy in predicting like counts for typical posts, which comprise the majority of Instagram content and fall within the 0-10,000 likes range.

key features - average like count per user, comment count, and timestamps - combined with log transformation to handle data skewness, resulted in a significant improvement in prediction accuracy, reducing our loss from 2667 to 597. While the model occasionally struggles with predicting exact numbers for viral content (45,000-60,000 likes range), this limitation is understandable given the unpredictable nature of viral success on social media.

## Responsibilities
Aksel Dindisyan, Muhammed Burak Gülümser
- Mainly focused on the regression task
- Explored the dataset and extracted useful features
- Initially implemented Random Forest and kNN models
- Afterwards switched to Gradient Boost for its effectiveness and efficiency
- Implemented Log Transformations to surmount the imbalance in the like count data
- Wrote down the findings in the report

İbrahim Başar Demir, Ahmet Coşkun:
- Mainly focused on the classification task
- Explored the dataset and extracted useful features
- Experimented with various approaches for the task before ending up with the final model
- Experimented with different ensemble approaches to see which model can achieve the highest accuracy
- Wrote down the findings in the report

Keep in mind that the members participated in both tasks, and not only on their focused task.

