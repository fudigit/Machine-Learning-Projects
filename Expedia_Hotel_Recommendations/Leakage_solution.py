# get all indices, groupby the columns to match
match_col = [ 'user_location_country',
       'user_location_region', 'user_location_city',
       'orig_destination_distance', 'srch_destination_id']

cluster_dict = train.groupby(match_col)

# find_exact_match function: given a row from test set & the matching columns, return the list of cluster(or [])
def find_exact_match(test_row, match_col):
    # create a tuple with the column values, since dict with multiindex takes tuples only
    index = tuple([test_row[t] for t in match_col])
    try:
        group = cluster_dict.get_group(index)
        return list(set(group['hotel_cluster']))
    # base class
    except Exception:
        return []


# for each row, find the match, append!
pred_labels = []
for i in range(test.shape[0]):
    clusters = find_exact_match(test.iloc[i], match_col)
    pred_labels.append(clusters)
    
