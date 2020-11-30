# Lyrics Generator

This is a Lyrics Generation Model trained built using LSTMs. The model have been trained on lyrics of different singers. The predict functions needs you to provide a root word or words and it predicts/ generates lyrics according to that root word.

## Dataset
The dataset used for training the model was present in the form of a txt file which contained the lyrics of some songs of various artists. 
This repository also contains lyrics data of several artists in case anyone wants to train the modeland generate different models. Go to [this folder.](https://github.com/ishantjuyal/Lyrics-Generator/tree/master/Lyrics%20data%20grouped%20by%20artists)

## Performance

Single line lyrics predicted after training on Taylor Swift lyrics. See the notebook [here.](https://github.com/ishantjuyal/Lyrics-Generator/blob/master/Taylor%20Swift%20Lyrics%20Prediction.ipynb)

![alt text](https://github.com/ishantjuyal/Word-Prediction/blob/master/Demo/Lyrics%201.png?raw=true)

Multiple lines of lyrics generated after training on Taylor Swift lyrics. See the notebook [here.](https://github.com/ishantjuyal/Lyrics-Generator/blob/master/Taylor%20Swift-%20Multiple%20Sentences%20Lyrics%20Generator%20.ipynb)

![alt text](https://github.com/ishantjuyal/Lyrics-Generator/blob/master/Demo/multiple_lines.png)

## Reference

[Kaggle Dataset](https://www.kaggle.com/PromptCloudHQ/taylor-swift-song-lyrics-from-all-the-albums)
