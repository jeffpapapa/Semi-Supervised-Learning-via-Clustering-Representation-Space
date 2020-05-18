# Semi Supervised Learning via Clustering Representation Space--- MCMC (Maximum Cluster Margin Classifier) in [pytorch](https://pytorch.org/)
### License
author : Yen-Chieh, Huang
Please cite `my paper(working now)` if you use this code or method.

## toy problems
### two-half moon
Four green points are the labeled points, and the rest of points are unlabeled points.
The figure shows the learning process by MCMC model.
![](https://i.imgur.com/eyDyYKP.gif)


## MNIST dataset
### Our result
![](https://i.imgur.com/b3JdWuJ.png)


### Training MCMC for semi-supervised learning
Use `main.py` to train semi-supervised learning with MCMC for MNIST dataset. Run `python main.py` to start training MNIST dataset with default 100 labeled samples, 59900 unlabeled samples and 10000 testing data. **Please download [MNIST dataset](https://www.kaggle.com/oddrationale/mnist-in-csv) with csv file first, then put training data and testing data in the same path.** Change parameters yourself for your own personal needs. Example usage : 
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
Note: you may also apply my loss function to your personal semi-supervised learning task.

### Requirements
These are our main enviorments to run the above:
* torch==1.0.1.post2
* torchvision==0.2.2.post3
* tqdm==4.40.0


## Citation
