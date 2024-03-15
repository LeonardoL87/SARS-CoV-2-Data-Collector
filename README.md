# Análisis de Datos de COVID-19

Este repositorio contiene un script de Python diseñado para analizar y visualizar datos de COVID-19 de Our World in Data. El script proporciona funciones para cargar los datos, realizar análisis exploratorios, visualizaciones y análisis de componentes principales (PCA) para países y continentes específicos.

## Requisitos

- Python 3
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Statsmodels

## Uso

1. Clona este repositorio en tu máquina local:

    ```bash
    git clone https://github.com/tu_usuario/analisis-covid19.git
    ```

2. Navega al directorio del repositorio:

    ```bash
    cd analisis-covid19
    ```

3. Ejecuta el script principal:

    ```bash
    python main.py
    ```

4. Selecciona las opciones disponibles en el menú para cargar los datos, realizar análisis exploratorios, visualizaciones y análisis de componentes principales.

## Funciones Principales

### `GetDataSets(Ploting=False, date)`

- Descarga los datos de Our World in Data sobre COVID-19.
- Filtra los datos según la fecha especificada.
- Retorna un DataFrame con los datos procesados.

### `SplitFitIntervals(Data)`

- Divide los datos en intervalos para ajustar el modelo de series temporales.
- Retorna un DataFrame con los intervalos de cada país.

### `PrincipalComp(Data, location)`

- Realiza un análisis de componentes principales (PCA) sobre los datos.
- Retorna un DataFrame con las componentes principales y gráficos de varianza explicada.

### `seasonality(Data, year, location, attributes)`

- Visualiza la estacionalidad de los atributos especificados a lo largo de los meses.
- Genera gráficos de caja para cada atributo.

### `Distribution(Data, year, atribute)`

- Visualiza la distribución de los atributos especificados por país.
- Genera gráficos de distribución conjunta utilizando Seaborn.

### `PlotCountryDynamics(Data, Country)`

- Visualiza la dinámica de los casos, muertes y recuperaciones para un país específico.
- Genera gráficos de líneas para los datos seleccionados.

## Contribuciones

Las contribuciones son bienvenidas. Si tienes sugerencias de mejoras, por favor, crea un issue o envía un pull request.

## Licencia

Este proyecto está bajo la licencia [MIT](LICENSE).
