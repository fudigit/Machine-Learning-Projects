'''
Build Binary Classifier for each hotel cluster
1. Loop through each hotel cluster.
  1. Create the new binary "label" for each row, indicates whether the row is in the cluster, or not.
  2. Train binary classifier for each of the K-fold and make predictions, extract the probablities of the cluster is "booked"
  3. Append the probabilities for a cluster to the whole probability list
2. Convert the whole probability list to data frame, transpose, add the column headers
3. For each row, find the 5 largest probabilities, and assign the hotel_cluster value as predictions
4. Compute accuracy via mapk
'''

9.21
Finished the main portion of the project. The best 2 way to make predictions are not machine learning, but:
  1. Use data leakage! The city and user_location_city and orig_destination_distance defines the hotel cluster with almost a 100% accuracy
    - The orig_destination_distance is determined only after hotel cluster is selected, this data shall not be in the trian set!
  2. Use the most popular 5 clusters for each srch_destination_id
  
之前airbnb花了不少时间研究这个项目，现在才发现基本没有用，因为没有用到ML。之前有很多东西看不懂，其实就是利用data leakage和most common的简单思想。
所以，仔细看数据本身，搞明白数据里的逻辑关系是最重要的！

