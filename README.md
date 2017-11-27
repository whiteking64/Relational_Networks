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
	1. Shape of the object which is closest to the certain colored object
	2. Shape of the object which is furthest to the certain colored object
	3. Number of objects which have the same shape with the certain colored object


## Requirements
+ Python 3
+ Numpy
+ Pytorch
+ OpenCV


## File details
There are 2 types of implementation: ```clevr_model``` and ```strict_model```.

### Directory info & details
```
Relational_Networks
  ├ README.md
  ├ clevr_model (Run with command line)
  │  ├ main.py (Load dataset, train, and test)
  │  ├ gen_dataset.py (generate Sort-of-CLEVR dataset)
  │  ├ model.py (Classes of CNN, RN, CNN_MLP, definitions of train and test)
  │  ├ data
  │  │  └ sort-of-clevr.pickle (where the dataset is stored)
  │  └ model
  │     ├ epoch_20_saved.pth (where learned models are saved)
  │     └ ...
  └ strict_model (Run with Jupyter notebook)
     └ (Same structure as above)
```


### Difference between files of ```clevr_model``` and ```strict_model```

|   | clevr_model | strict_model |
|:-------------:|:----------------------|:---------------------- |
| RN | Implemented the CLEVR model configuration at p.6 with minor changes | Strictly implemented the original paper at p.12 (number of layers, parameters, etc.) |
| comment out | English | Japanese (Mostly) |
|   | little comments | detailed comments |
| supplementary  | ○ | × |

※ Python version with CLEVR model takes much less computation time.  
※ Supplementary is explanation of the codes.


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

