# Pytorch_InceptionTime

This is Pytorch implementation of a 1D-CNN Time-series classification model [InceptionTime](https://github.com/hfawaz/InceptionTime)


I test my code with the environment below:
### Environment 
python == 3.5  
pytorch == 1.1.0  
scikit-learn == 0.21.3

His results on 128 UCR datasets can be found at [here](https://github.com/hfawaz/InceptionTime/blob/master/results-inception-128.csv)

The Accuracy comparison on UCR 128 datasets with one time run of our code is below [image](https://github.com/Wensi-Tang/Pytorch_InceptionTime/files/6374951/en.pdf)
#### Some datasets' results are significantly different—the difference between data preprocessing causes that.
For example, for PLAID dataset, the author used a z-normalized PLAID dataset, but we keep the original from the UCR archive.

![en-page-001 (1)](https://user-images.githubusercontent.com/61366756/116044975-e74f7380-a6b4-11eb-8490-683544ba0e2f.jpg)


