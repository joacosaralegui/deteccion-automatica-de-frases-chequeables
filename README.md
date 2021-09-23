# Trabajo Final de carrera Ingenieria de Sistemas

## Detección automática de frases chequeables
Este trabajo fue realizado cómo Trabajo Final para la carrera de Ingeniería de Sistemas de la Universidad Nacional del Centro de la Provincia de Buenos Aires. 

## Instalación
Requiere Python 3 y [VirtualEnv](https://help.dreamhost.com/hc/es/articles/115000695551-Instalar-y-usar-virtualenv-con-Python-3).
```
virtualenv env -p python3
source env/bin/activate
pip install -r requirements.txt
```

## Uso
Para entrenar los diferentes pipelines
```
python src/train_model.py
```
Para utilizar el clasificador en producción o probar con diferentes frases sueltas ver src/classifier.py
