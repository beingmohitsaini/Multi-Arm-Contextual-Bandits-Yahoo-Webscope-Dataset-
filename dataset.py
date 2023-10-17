"""
A line of data from yahoo events:
1241160900 109513 0 |user 2:0.000012 3:0.000000 4:0.000006 5:0.000023 6:0.999958 1:1.000000 |109498 2:0.306008 3:0.000450 4:0.077048 5:0.230439 6:0.386055 1:1.000000 |109509 2:0.306008 3:0.000450 4:0.077048 5:0.230439 6:0.386055 1:1.000000 [[...more article features omitted...]] |109453 2:0.421669 3:0.000011 4:0.010902 5:0.309585 6:0.257833 1:1.000000
Each line represents an event(logged). The various entries in the data can be broken down into components as:
1. First number represents the time stamp,1241160900 in the above line.
2. Second entry is the article index shown to user relative to the pool index
3. The next entry represents whether the user clicked on that article (1) or not (0)
4. Next 6 entries are the features of the user.
5. From 10th Entry onwards starts the info about articles, first article id is there followed by 6 entries representing the features of the article

Each article has 7 columns (article id + 6 features)
Therefore number_of_columns-10 % 7 = 0 for an event to be a valid event , if this condotion does not hold true we skip that event in our evaluation.
"""

import numpy as np # importing numpy library
import fileinput # importing fileinput library to read the data file

def get_events(filenames): 
    """
    This function reads a stream of events from the given files.
    
    Parameters passed:
    ----------
    filenames :List of filenames
    
    What it stores:
    -------    
    articles : [article_ids] - a list of article ids as they appear in the datset event by evebt
    a_features : [[article_1_features] .. [article_n_features]] -- a matrix to store features of all unique articles in the dataset,event by event
    events : [
                 0 : displayed_article_index (relative to the pool),
                 1 : user_click, denoted by 0 or 1
                 2 : [user_features],
                 3 : [pool_indexes] , indexes of articles for that particular event
             ]
    """

    global articles, a_features, events, n_arms, n_events
    articles = []
    a_features = []
    events = []

    skipped = 0 # counter to store number of bad or skipped events

    with fileinput.input(files=filenames) as f: # opening the data file as f
        for line in f: # traverse through each line/event in the dataset
            col = line.split() #this stores all the entries in the line
            if (len(col) - 10) % 7 != 0:
                skipped += 1
            else:
                pool_idx = [] # to store article indices for the particular event
                pool_ids = [] # ids of article in the pool
                for i in range(10, len(cols) - 6, 7): #starts with first article and goes on to the first column of the last article present in the line
                    id = col[i][1:] #get the article id from the event
                    if id not in articles:
                        articles.append(id)
                        a_features.append([float(x[2:]) for x in col[i + 1: i + 7]]) #get features of articles and append it to a_features
                    pool_idx.append(articles.index(id)) #get the index of the article id in the articled list and append it to pools_idx
                    pool_ids.append(id) #append the article id to the pool_ids

                events.append(
                    [
                        pool_ids.index(col[1]), #col[1] represents the article id which was shown to the user,we get index of that id in the pool
                        int(col[2]), #col[2] represents the user_click
                        [float(x[2:]) for x in col[4:10]], # these are user features for that event
                        pool_idx,
                    ]
                )
    a_features = np.array(a_features) #converting article features into numpy array
    n_arms = len(articles) #total number of unique articles
    n_events = len(events) #total no of events
    print("There are "n_events, "with", n_arms, "unique articles")
    if skipped != 0:
        print("Total number of Skipped events:", skipped)

def max_articles(n_articles):
    """
    Reduces the number of articles to the threshold provided.
    Therefore the number of events will also be reduced.

    Parameters passed:
    ----------
    n_articles :number of max articles after reduction
    """

    global articles, a_features, events, n_arms, n_events
    assert n_articles < n_arms #asserting that the reduced number of articles is less than the number of articles already present(n_arms)

    n_arms = n_articles
    articles = articles[:n_articles] # select reduced number of ids from all article ids
    a_features = a_features[:n_articles] # select features of reduced number of articles from features of all articles

    for i in reversed(range(len(events))):
        displayed_pool_idx = events[i][0]  # index relative to the pool
        displayed_article_idx = events[i][3][displayed_pool_idx]  # index relative to the articles

        if displayed_article_idx < n_arms:
            events[i][0] = displayed_article_idx
            events[i][3] = np.arange(0, n_arms)  # pool = all available articles
        else:
            del events[i]

    n_events = len(events)
    print("New number of events:", n_events)
