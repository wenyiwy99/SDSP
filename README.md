# SDSP
An official source code for paper Measure Domain's Gap: A Similar Domain Selection Principle for Multi-Domain Recommendation

## Environment
* python
* pytorch
* torchvision
* pandas
* tqdm
* scikit-learn

If you use anaconda, you can run the following command:
```
conda create -n rec python=3.8
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pandas tqdm scikit-learn
```

## Run the model
```
python run_movielens.py
python run_amazon.py 
python run_douban.py 
```
