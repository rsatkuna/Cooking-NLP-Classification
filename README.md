Project 4: What's Cooking

My team and I chose the Kaggle competition, [What's Cooking?](https://www.kaggle.com/c/whats-cooking). We identified a relevant NLP problem involving classification. Specifically, we were interested in predicting the type of cuisine based on a common list of ingredients.Additionally, we performed some EDA, and fit and evaluated two models on the chosen dataset.

## Produced by:
- Kate Crawford
- Revathi Satkuna
- Amanda Walsh
- Preet Sekhon

## Datasets

- [What's Cooking?](https://www.kaggle.com/c/whats-cooking)

## Problem Statement (from Kaggle)

If you're in Northern California, you'll be walking past the inevitable bushels of leafy greens, spiked with dark purple kale and the bright pinks and yellows of chard. Across the world in South Korea, mounds of bright red kimchi greet you, while the smell of the sea draws your attention to squids squirming nearby. India’s market is perhaps the most colorful, awash in the rich hues and aromas of dozens of spices: turmeric, star anise, poppy seeds, and garam masala as far as the eye can see.

Some of our strongest geographic and cultural associations are tied to a region's local foods. This playground competitions asks you to predict the category of a dish's cuisine given a list of its ingredients. 

## My Contributions

During the cleaning process, one of my team members recommended that we determine if there are any duplicate rows. We found that there only 20 unique lists of ingredients in the entire dataset. This meant that recipes from different cuisines required the same ingredients. From this discovery, we chose to focus on retaining as much information about each ingredient at possible.

If an ingredient had more than one entity (i.e. Olive Oil), it may have lost it's value if it wasn't accounted for during vecotrizing. So, I created a class that would return a new vocabulary to input into our vectorizer. 

```Python
class ProcessedFoods:
    def __init__(self, df, column):
        self.df = df
        self.column = column
        
    # Add column with character count
    def get_char_count(self, new_column='char_count'):
        self.df[new_column] = self.df[self.column].apply(lambda x: sum([len(item) for item in x]))
        return self.df[new_column]
    
    # Add column with ingredient count 
    def get_ingredient_count(self, new_column='ingredient_count'):
        self.df[new_column] = self.df[self.column].apply(lambda x: len(str(x).split(',')))
        return self.df[new_column]
    
    # Create a new column where values are list type
    def convert_to_list(self, new_column='ingredient_list'): 
        self.df[new_column] = [i[2:-2].split("', '") for i in self.df[self.column]]
        return self.df[new_column]
    
    # Make vocabulary list of unique values in each row's list
    def make_vocabulary(self, column):
        list_name = []
        
        # prefer to use df from convert_to_list
        for i in range(len(column)):
            for item in column[i]:
                if item not in list_name:
                    list_name.append(item)
                        
        # pickle for your pleasure
        with open('../data/word_list.pkl', 'wb') as pickle_in:
            pickle.dump(list_name, pickle_in)
        
        # set equal to vocabulary for vectorizer
        return list_name
```

This allowed to run just a few lines of code to create a cusotm vocabulary list and do feature engineering. If I could return to this project, I would have included this code in importing this class from a separate python file. I am also interested in improving the efficiency of this code and generalizing it for future use.
