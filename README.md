# Relational Networks -- Sort-of-CLEVR --
+ Pytorch implementation of Relational Networks - [A simple neural network module for relational reasoning](https://arxiv.org/abs/1706.01427)  
+ Implemented & tested on Sort-of-CLEVR task.

## Sort-of-CLEVR
+ Simplified version of [CLEVR](http://cs.stanford.edu/people/jcjohns/clevr/) dataset
+ Composed of 10000 images and 20 questions (10 relational and 10 non-relational questions) per each image
+ One image contains 6 randomely chosen shapes (square or circle) with different colors (red, green, blue, orange, gray, yellow)
+ Non-relational and relational questions fall into 3 subtypes respectively:  
	[Non-relational]
	1. Shape of certain colored object
	2. Horizontal location of certain colored object : whether it is on the left side of the image or right side of the image
	3. Vertical location of certain colored object : whether it is on the upside of the image or downside of the image
	  
	[relational]
	1. Shape of the object that is closest to the certain colored object
	2. Shape of the object that is furthest to the certain colored object
	3. Number of objects that have the same shape as the certain colored object


## Requirements
+ Python 3
+ Numpy
+ Pytorch
+ OpenCV


## File details
There are 2 folders: ```Python_ver``` and ```iPython_ver```.

### Directory info & details
```
Relational_Networks
  ├ README.md
  ├ Python_ver (Run with command line)
  │  ├ main.py (Load dataset, train, and test)
  │  ├ gen_dataset.py (generate Sort-of-CLEVR dataset)
  │  ├ model.py (Classes of CNN, RN, CNN_MLP, definitions of train and test)
  │  ├ data
  │  │  └ sort-of-clevr.pickle (where the dataset is stored)
  │  └ model
  │     ├ epoch_20_saved.pth (where learned models are saved)
  │     └ ...
  └ iPython_ver (Run with Jupyter notebook)
     └ (Same structure as above)
```

※ The reference code[1] (Sort-of-CLEVR dataset with CLEVR model) takes much less computation time.  
※ Comment outs are mostly Japanese, but large part of my code is based on reference[1].


## Usage
Generate the dataset by:

```
$ python gen_dataset.py
```

Then, train & test by:

```
$ python main.py
```


## References
1. [relational-networks](https://github.com/kimhc6028/relational-networks)

