# Técnicas de Clusterización Aplicadas en la Selección del Periodo de Indicadores Técnicos

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
La idea general es aplicar clusterización tipo kmeans para agrupar periodos de un indicador y sobre dichos grupos seleccionar el periodo de indicador que tenga mayor correlación con la variable a predecir. Este permite, por un lado, mitigar la multicolinealidad, y por otro lado, conservar los periodos del indicador técnico que aporten mayor poder predictivo.
Por otro lado, desde un enfoque descriptivo se usa la técnica de clusterización para estudiar posibles indicadores técnicos que permitan filtrar regímenes de mercado.
Se complementa el estudio con la técnica de PSI (Population Stability Index), que permite evaluar si una feature ha presentado un cambio abrupto en sus propiedades estadísticas o drift en la ventana de train y test. 

## Instalación

## Requisitos

Para la ejecución del programa es necesario instalar la version de python 3.11.9, vscode y los paquetes del archivo requirements.txt

Instale [python](https://www.python.org/downloads/) y [vscode](https://code.visualstudio.com/download)

## Uso

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

Este proyecto fue una gran oportunidad para combinar mis habilidades en data science con mi interés en mejorar la movilidad urbana. Utilizando el conjunto de datos de Uber en Nueva York, he diseñado una red de transporte básica que busca optimizar el servicio en áreas de alta demanda.

Principales Aprendizajes:

- Eficiencia del k-means: El método k-means resultó ser la herramienta más efectiva para identificar áreas clave para nuevas paradas de transporte. Este método fue directo y produjo clusters claros y manejables, facilitando la siguiente fase de diseño de la red.

- Implementación del algoritmo de Kruskal: Aplicar Kruskal para conectar estas nuevas paradas fue una decisión técnica clave que ayudó a minimizar los costos y maximizar la eficiencia de la red propuesta. Fue satisfactorio ver cómo una técnica teórica se traduce en aplicaciones prácticas que podrían, en teoría, implementarse en la vida real.

## Autor

Jhon Jairo Realpe

jhon.td.03@gmail.com

