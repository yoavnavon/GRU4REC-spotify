# GRU4REC-spotify

This project aims to tackle the problem of session based recommendation. Unlike traditional
recommender systems, where the data of each user is available, and it's possible to build a 
profile of the user, in session based recommendation, we try to give recommendations based
only on picks in the current session. To solve this problem, we are going to use the [GRU4REC](http://arxiv.org/abs/1511.06939)
model for recommendation, and expand it with new features.

The original model was trained and evaluated on a dataset with only 37.000 items to be recommended, so we are going to put it to the limit by evaluating it in a dataset belonging to [spotify](https://www.aicrowd.com/challenges/spotify-sequential-skip-prediction-challenge). This dataset has more than 500.000 items to recommend, making the data much more sparse and difficult to predict.

The dataset has information for each item (songs) that can be recommended, so content-based recommendation is possible. To do so, we are going to try embedding layers that are going to encode content vectors of each item, and these will be the input of the GRU4REC model. In contrast, the original model only uses one-hot representations for items.

The project was created in the context of the course "Recommender Systems", and more complete detail of the project can be found in the final [paper](reports/paper.pdf).
## Dependencies
- Python 3.x
- keras + tensorflow
- pandas
- tqdm
- numpy
- scikit-learn

## Data Download

#### Sessions

```
>> wget https://recsys-spotify.s3.amazonaws.com/training_subsample_1.tar.gz
>> tar -xzf training_subsample_1.tar.gz -C data/
>> mv data/log_3_20180827_000000000000.csv.gz data/training/log_3_20180827_000000000000.csv.gz
```

#### Track Features

```
>> wget https://os.zhdk.cloud.switch.ch/swift/v1/crowdai-public/spotify-sequential-skip-prediction-challenge/20181120_track_features.tar.gz
>> tar -xzf 20181120_track_features.tar.gz -C data/
  
```

## Preprocessing

```
>> python preprocess.py
```

## Training

```
>> python run.py
```



