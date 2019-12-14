# Proyecto Final Sistemas Recomendadores
Este es el repositorio del proyecto final del curso Sistemas Recomendadores. El proyecto consiste en un sistema recomendador basado en sesiones. El modelo está basado en el propuesto en "Session-based Recommendations With Recurrent Neural Networks" (http://arxiv.org/abs/1511.06939). El código está basado en la implementación en https://github.com/pcerdam/KerasGRU4Rec. En el notebook Spotify GRU Example se muestra un ejemplo de entrenamiento. El modelo se encuentra detallado en el [paper](Paper_proyecto_Recsys.pdf).
## Dependencias
- python 3.x
- Keras + Tensorflow
- pandas
- tqdm
- numpy
- scikit-learn

## Descarga

#### Sesiones

```
>> wget https://recsys-spotify.s3.amazonaws.com/training_subsample_1.tar.gz
>> tar -xzf training_subsample_1.tar.gz -C data/
>> mv data/log_3_20180827_000000000000.csv.gz data/training/log_3_20180827_000000000000.csv.gz
```

#### Features de Tracks

```
>> wget https://os.zhdk.cloud.switch.ch/swift/v1/crowdai-public/spotify-sequential-skip-prediction-challenge/20181120_track_features.tar.gz
>> tar -xzf 20181120_track_features.tar.gz -C data/
  
```

## Preprocesamiento

```
>> python preprocess.py
```

## Entrenamiento

Los parámetros se pueden modificar directamente en el archivo `run.py`. Alternativamente, si se utiliza un jupyter notebook se puede importar la función `train_model`dentro de `train.py`.
```
>> python run.py
```



