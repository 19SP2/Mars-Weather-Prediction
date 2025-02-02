## Mars Weather Prediction using Data from NASA’s InSight Mars Lander
NASA’s InSight Mars lander takes continuous weather measurements (temperature, wind, pressure) on the surface of Mars at Elysium Planitia, a flat, smooth plain near Mars’ equator. This API provides per-Sol summary data for each of the last seven available Sols (Martian Days).

The API doc: https://api.nasa.gov/assets/insight/InSight%20Weather%20API%20Documentation.pdf

Check out NASA's Open APIs: https://api.nasa.gov/?ref=freepublicapis.com

## Problem Understanding and Definition
If we are to send humans on Mars we must know its weather conditions.

Predicting weather on Mars has many advantages, it helps in understanding the planets atmosphere and its significance in the solar system. It is very crucial to understand the weather of the planet for future exploration. It also helps in planning robotic missions as well.

NASA's InSight Lander launched on May 5, 2018 collected data from the planets atmosphere. This Machine Learning Model uses supervised learning to predict data collected from the lander.

Due to the challenges of the mission, very little data was collected by the lander, therefore creating a model with this data will result in a rather inaccurate model. But with more data the model will become accurate.
