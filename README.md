# Clustering Structure Embedding Space for Semi Supervised Learning --- MCMC (Maximum Cluster Margin Classifier) in [pytorch](https://pytorch.org/)
### License

## MNIST dataset
Please download [MNIST dataset](https://www.kaggle.com/oddrationale/mnist-in-csv)
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

