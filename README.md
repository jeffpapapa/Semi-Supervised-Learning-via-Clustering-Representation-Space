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
Use `main.py` to train semi-supervised learning with MCMC for MNIST dataset. Run `python main.py --dataroot=mnist_data --dataset=mnist --method=mcmc` to start training MNIST dataset with default 100 labeled samples, 59900 unlabeled samples and 10000 testing data. **Please download [MNIST dataset](https://www.kaggle.com/oddrationale/mnist-in-csv) with csv file first, then put training data and testing data in the same path.** Change parameters yourself for your own personal needs. Example usage : 
```
python main.py --dataroot=mnist_data --dataset=mnist --method=mcmc \
  --num_per_class=10 \
  --num_epochs=10 \
  --epoch_decay_start=3 \
```
Note: you may also apply my loss function to your personal semi-supervised learning task.

### Requirements
These are our main enviorments to run the above:
* torch==1.0.1.post2
* torchvision==0.2.2.post3
* tqdm==4.40.0

### Try your own data
If you want to apply MCMC for your own data set, you need the following process:
1. retrun embedding space z from your model(network), check our example `model.py`
2. import our loss functions `from MCMC_loss import DBI, margin`
3. remember to load your unlabeled data(in my paper, I use unlabeled+labeled here) for every iteration
4. check our code in `main.py`, find function `train_mcmc`, compute our loss followed by our example code
5. feed labeled data and unlabeled data(in my paper, I use unlabeled+labeled here) into our loss functions followed by our code
6. train your model!

### Notification
You can also modify more parameter settings such as iterations per epoch, learning rate,..., etc. from the top of main.py

## Citation
