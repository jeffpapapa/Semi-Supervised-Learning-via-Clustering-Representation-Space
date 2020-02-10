# Clustering Structure Embedding Space for Semi Supervised Learning --- MCMC (Maximum Cluster Margin Classifier) in [pytorch](https://pytorch.org/)
### License

## MNIST dataset
Use 'main.py' to train semi-supervised learning with MCMC for MNIST dataset. Run `python main.py` to start training MNIST dataset with default 100 labeled samples, 59900 unlabeled samples and 10000 testing data. _Please download [MNIST dataset](https://www.kaggle.com/oddrationale/mnist-in-csv) with csv file, put training data and testing data in the same path._ Change parameters yourself for your own personal needs. Example usage : 
```
python main.py --path mnist-in-csv \
  --experiment 10 \
  --labelnum 100 \
  --clusterdim 300 \
  --kmargin 10 \
  --batch 200 \
  --test_batch 100 \
  --label_batchsize 50 \
  --learning_rate 0.001 \
  --epoch 30 \
```

