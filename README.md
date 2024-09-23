# Técnicas de Clusterización Aplicadas en la Selección del Periodo de Indicadores Técnicos

![candles fig](https://github.com/jhontd03/FeaturesCluster/blob/master/img/candles.png "candles_fig")

## [Contenido](#Contenido)

- [Introducción](#Introduccón)
- [Instalación](#Instalación)
- [Requisitos](#Requisitos)
- [Uso](#Uso)
- [Estructura del repositorio](#Estructura-del-repositorio)
- [Resultados](#Resultados)
- [Autor](#Autor)

## Introducción

Este repositorio se enfoca en la aplicación de técnicas de clusterización para seleccionar el periodo de indicadores técnicos, que son la base para generar modelos de machine learning aplicados al trading algorítmico.
La idea general es aplicar clusterización tipo K-means para agrupar periodos de un indicador y sobre dichos grupos seleccionar el periodo de indicador que tenga mayor correlación con la variable a predecir. Este permite, por un lado, mitigar la multicolinealidad, y por otro lado, conservar los periodos del indicador técnico que aporten mayor poder predictivo.
Por otro lado, desde un enfoque descriptivo se usa la técnica de clusterización para estudiar posibles indicadores técnicos que permitan filtrar regímenes de mercado.
Se complementa el estudio con la técnica de PSI (Population Stability Index), que permite evaluar si una feature ha presentado un cambio abrupto en sus propiedades estadísticas o drift en la ventana de train y test. 

## Instalación

### Requisitos

Para la ejecución del programa es necesario instalar la version de python 3.11.9, vscode y los paquetes del archivo requirements.txt

Instale [python](https://www.python.org/downloads/) y [vscode](https://code.visualstudio.com/download)

### Uso

Clone el presente repositorio cree un entorno virtual, instale las librerias y ejecute el notebook con vscode.

```
git clone https://github.com/jhontd03/FeaturesCluster.git
cd FeaturesCluster
conda create -n FeaturesCluster python=3.11.9 
activate FeaturesCluster
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Estructura del repositorio

El árbol de directorios del repositorio es el siguiente:
```
.
│   basic_features.py
│   data_labeling.py
│   helper_functions.py
│   indicators.py
│   LICENSE
│   nancorrmp.py
│   period_indicator_selection.ipynb
│   README.md
│   study_cluster_indicator.ipynb
│   __init__.py
```

## Conclusiones

- La clusterización como herramienta para selección de periodos de indicadores técnicos: El uso de técnicas de clusterización, como K-means, facilita la selección de los periodos de indicadores técnicos más relevantes. Al agrupar periodos similares y seleccionar aquellos con mayor correlación con la variable objetivo, se mitiga la multicolinealidad y se preservan solo las características que aportan mayor poder predictivo al modelo.

- Identificación de regímenes de mercado: El uso de clusterización con un enfoque descriptivo permite identificar patrones dentro de los indicadores técnicos que revelan posibles regímenes de mercado. Este análisis descriptivo puede ayudar a los traders a adaptar sus estrategias según las condiciones cambiantes del mercado.

- Control del drift en las features mediante PSI: La inclusión del Population Stability Index (PSI) es fundamental para monitorear la estabilidad de las características a lo largo del tiempo. Esto permite detectar cambios estadísticos significativos o drift entre las ventanas de train y test, garantizando que los modelos se mantengan efectivos y no se degraden debido a cambios en las propiedades de los datos.

## Autor

Jhon Jairo Realpe

jhon.td.03@gmail.com

