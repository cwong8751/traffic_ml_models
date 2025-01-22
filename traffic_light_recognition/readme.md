# traffic light recognition 

this is a sub project that identifies traffic light based on images and videos. done with a CNN. 



## data structure
```models/``` contains models in both h5 and tflite formats
```dataset/``` contains the datasets
```x_test.npy x_train.npy y_test.npy y_train.npy``` train test splits from the jupyter notebook
```camrecog.py``` python file that triggers camera to do real time identification, requires model.h5 in the models folder.

## datasets

https://zenodo.org/records/12706046 - bosch small traffic light dataset

https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset - LISA traffic light dataset

https://github.com/DeepeshDongre/Traffic-Light-Detection-LaRA?tab=readme-ov-file - LARA dataset



## installing tensorflow
tensorflow doesn't seem to work on 3.12 or 3.11. we use 3.8

https://github.com/ChaitanyaK77/Initializing-TensorFlow-Environment-on-M3-M3-Pro-and-M3-Max-Macbook-Pros.?tab=readme-ov-file
