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

Este respositoro se enfoca en 

## Instalación

## Requisitos

Para la ejecución del programa es necesario instalar la version de python 3.11.9, vscode y los paquetes del archivo requirements.txt

Instale [python](https://www.python.org/downloads/) y [vscode](https://code.visualstudio.com/download)

## Uso

Clone el presente repositorio cree un entorno virtual, instale las librerias y ejecute el notebook con vscode.

```
conda create -n FeaturesCluster python=3.11.9 
activate FeaturesCluster
python -m pip install --upgrade pip
pip install -r requirements.txt
git clone https://github.com/jhontd03/RedesTransporte.git
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
