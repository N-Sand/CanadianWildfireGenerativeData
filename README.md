# CanadianWildfireGenerativeData
Motivated by the record uncontrolled fires in Canada in 2023.
We look at data from the NASA Modis satellite and perform some analysis. We implement a simple variational autoencoder with the hopes of generating new data similar to historic Canadian fire data.. The idea is that one could use these hypothetical scenarios to better prepare for possible future fires.

Data source: NASA Modis Satellite: https://firms.modaps.eosdis.nasa.gov/download/
VAE code inpired and modified from: Alexander Van de Kleut (link: https://avandekleut.github.io/vae/) (A fantastic read)

Requirements:
numpy, datetime
scipy, sklearn
matplotlib
pandas, geopandas
torch
