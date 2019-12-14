# Proyecto Final Sistemas Recomendadores

## Dependencias
- python 3.x
- Keras + Tensorflow
- pandas
- tqdm

## Descarga
#### Sesiones
```
>> wget https://recsys-spotify.s3.amazonaws.com/training_subsample_1.tar.gz
>> tar -xzf training_subsample_1.tar.gz -C data/
>> mv data/log_3_20180827_000000000000.csv.gz data/training/log_3_20180827_000000000000.csv.gz
```
#### Features de Tracks
```
>> wget https://os.zhdk.cloud.switch.ch/swift/v1/crowdai-public/spotify-sequential-skip-prediction-challenge/20181120_track_features.tar.gz
>> tar -xzf 20181120_track_features.tar.gz -C data/
  
```
## Preprocesamiento
```
>> python preprocess.py
```

## Entrenamiento
```
>> python run.py
```



