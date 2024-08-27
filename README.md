# PopPBPK-RL_Bnzpril
Welome to PopPBPK-RL_Bnzpril! This repository showcases the codebase for RL model-informed precision of Benazepril in renal impaired patients

## Versions used 

- R: 4.3.2
- Python: 3.12.0

## Installation 

Clone this repository into your local machine

```
git clone git@github.com:lucis-e/PopPBPK-RL_Bnzpril.git
cd PopPBPK-RL_Bnzpril
```

>[!TIP]
>To run this project, ensure you have Python 3.12.0 and R 4.3.2 installed on your system. You can download Python from [python.org](https://www.python.org/downloads/release/python-3120/) and R from [cran.r-project.org](https://cran.r-project.org/bin/windows/base/). Additionally, we recommend installing RStudio for a more convenient development environment. You can download RStudio from 
[rstudio.com](https://www.rstudio.com/products/rstudio/download/).

## How to run this project?

Please follow these steps indicating the order in which each code script should be executed

### 1. Feature selection and virtual population stratification

:one:  <span style="color:blue"> **MLR analysis for feature selection** </span>

Run the R Markdown notebook to perform MLR and partial correlation coefficient analysis, discovering the anthropometric parameters that are most influential in PK profile prediction.

:two: **Virtual population stratification based on the extracted features**

Run the Python notebook to classify the virtual population into the defined strata based on the MLR-selected features and calculate the distribution of the PK parameters in each group.


### 2. RL agent

:three: 


## Data

The data employed in this project is available under formal request to: 
