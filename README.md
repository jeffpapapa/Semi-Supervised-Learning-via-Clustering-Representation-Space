# Clustering Structure Embedding Space for Semi Supervised Learning --- MCMC (Maximum Cluster Margin Classifier) in [pytorch](https://pytorch.org/)
### License

## MNIST dataset
### Training MCMC for semi-supervised learning
Use `main.py` to train semi-supervised learning with MCMC for MNIST dataset. Run `python main.py` to start training MNIST dataset with default 100 labeled samples, 59900 unlabeled samples and 10000 testing data. **Please download [MNIST dataset](https://www.kaggle.com/oddrationale/mnist-in-csv) with csv file, put training data and testing data in the same path.** Change parameters yourself for your own personal needs. Example usage : 
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

### Requirements
These are our main enviorments to run the above:
* torch==1.0.1.post2
* torchvision==0.2.2.post3
* tqdm==4.40.0


## Citation

