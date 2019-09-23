import os
cwd = os.getcwd()

import pandas as pd
train = pd.read_csv('train.csv')

test = train[:10]

match_col = [ 'user_location_country',
       'user_location_region', 'user_location_city',
       'orig_destination_distance', 'srch_destination_id']

## find match in test that's in train



groups = train[:20].groupby(match_col)


print(groups.get_group((66, 348, 48862, 2234.2641, 8250)))

for i in range(len(test)):
  index = tuple([v for v in test.iloc[i][match_col]])
  print(index)
  try:
    group = groups.get_group(index)
    match_cluster = list(set(group['hotel_cluster']))
    print(match_cluster)
  
  except Exception:
    print([])


print(groups.groups)
