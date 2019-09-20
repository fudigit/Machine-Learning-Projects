###
Build Binary Classifier for each hotel cluster
1. Loop through each hotel cluster.
  1. Create the new binary "label" for each row, indicates whether the row is in the cluster, or not.
  2. Train binary classifier for each of the K-fold and make predictions, extract the probablities of the cluster is "booked"
  3. Append the probabilities for a cluster to the whole probability list
2. Convert the whole probability list to data frame, transpose, add the column headers
3. For each row, find the 5 largest probabilities, and assign the hotel_cluster value as predictions
4. Compute accuracy via mapk
 
###
