import torch
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from ltr.loss import pointwise_loss, pairwise_loss, listwise_loss, compute_lambda_i
from .dataset import LTRData, QueryGroupedLTRData, qg_collate_fn
from .eval import evaluate_model


def train_batch(net, x, y, loss_fn, optimizer):
    optimizer.zero_grad()
    out = net(x)
    loss = loss_fn(out, y)
    loss.backward()
    optimizer.step()


# TODO: Implement this!
def train_pointwise(net, params, data):
    """
    This function should train a Pointwise network.

    The network is trained using the Adam optimizer


    Note: Do not change the function definition!


    Hints:
    1. Use the LTRData class defined above
    2. Do not forget to use net.train() and net.eval()

    Inputs:
            net: the neural network to be trained

            params: params is an object which contains config used in training
                (eg. params.epochs - the number of epochs to train).
                For a full list of these params, see the next cell.

    Returns: a dictionary containing: "metrics_val" (a list of dictionaries) and
             "metrics_train" (a list of dictionaries).

             "metrics_val" should contain metrics (the metrics in params.metrics) computed
             after each epoch on the validation set (metrics_train is similar).
             You can use this to debug your models.

    """

    val_metrics_epoch = []
    train_metrics_epoch = []
    optimizer = Adam(net.parameters(), lr=params.lr)
    loss_fn = pointwise_loss

    ### BEGIN SOLUTION
    # Step Tips:
    # Step 1: Create a DataLoader for the training data
    dataloader = DataLoader(
        LTRData(data, "train"), batch_size=params.batch_size, shuffle=True
    )

    # Step 2: Iterate over the data for the number of epochs
    for epoch in range(params.epochs):
        net.train()
        for x, y in dataloader:
            # Step 3: Iterate over each batch of data and use the train_batch function to train the model
            train_batch(net, x, y, loss_fn, optimizer)

        # Step 4: At the end of the epoch, evaluate the model on the data using the evaluate_model function (bot train and val)
        net.eval()
        train_metrics = evaluate_model(data, net, "train")
        val_metrics = evaluate_model(data, net, "validation")
        # Step 5: Append the metrics to train_metrics_epoch and val_metrics_epoch
        train_metrics_epoch.append(train_metrics)
        val_metrics_epoch.append(val_metrics)

    return {"metrics_val": val_metrics_epoch, "metrics_train": train_metrics_epoch}
    ### END SOLUTION


# TODO: Implement this!
def train_batch_vector(net, x, y, loss_fn, optimizer):
    """
    Takes as input a batch of size N, i.e. feature matrix of size (N, NUM_FEATURES), label vector of size (N), the loss function and optimizer for computing the gradients, and updates the weights of the model.
    The loss function returns a vector of size [N, 1], the same as the output of network.

    Input:  x: feature matrix, a [N, NUM_FEATURES] tensor
            y: label vector, a [N] tensor
            loss_fn: an implementation of a loss function
            optimizer: an optimizer for computing the gradients (we use Adam)
    """
    ### BEGIN SOLUTION
    # Step tips:
    # Step 1: Zero the gradients of the optimizer
    optimizer.zero_grad()
    # Step 2: Forward pass the input through the network using the net
    out = net(x)
    # Step 3: Compute the loss using the loss_fn
    loss = loss_fn(out, y)
    # Step 4: Backward pass to compute the gradients
    loss.backward()
    # Step 5: Update the weights using the optimizer
    optimizer.step()
    ### END SOLUTION


# TODO: Implement this!
def train_pairwise(net, params, data):
    """
    This function should train the given network using the pairwise loss

    Returns: a dictionary containing: "metrics_val" (a list of dictionaries) and
             "metrics_train" (a list of dictionaries).

             "metrics_val" should contain metrics (the metrics in params.metrics) computed
             after each epoch on the validation set (metrics_train is similar).
             You can use this to debug your models

    Note: Do not change the function definition!
    Note: You can assume params.batch_size will always be equal to 1

    Hint: Consider the case when the loss function returns 'None'

    net: the neural network to be trained

    params: params is an object which contains config used in training
        (eg. params.epochs - the number of epochs to train).
        For a full list of these params, see the next cell.
    """

    val_metrics_epoch = []
    train_metrics_epoch = []
    ### BEGIN SOLUTION
    # Step Tips:
    # Step 1: Create a DataLoader for the training data
    train_data = DataLoader(QueryGroupedLTRData(data, "train"), shuffle=True, collate_fn=qg_collate_fn)

    # Step 2: Create your Adam optimizer
    optimizer = Adam(net.parameters(), lr=params.lr)

    # Step 3: Iterate over the data for the number of epochs
    for epoch in range(params.epochs):
        net.train()

        # Step 4: Iterate over each batch of data and compute the scores using the forward pass of the network
        for batch in train_data:
            features_q_i, labels_q_i = batch
            features_q_i = torch.cat(features_q_i, dim=0)
            labels_q_i = torch.cat(labels_q_i, dim=0)
            
            optimizer.zero_grad()

            # Step 5: Compute the pairwise loss using the pairwise_loss function
            scores_q_i = net(features_q_i)
            loss_q_i = pairwise_loss(scores_q_i, labels_q_i)

            # Step 6: Compute the gradients and update the weights using the optimizer
            if loss_q_i is not None:
                loss_q_i.backward()
                optimizer.step()

        # Step 7: At the end of the epoch, evaluate the model on the data using the evaluate_model function (both train and val)
        net.eval()
        train_eval = evaluate_model(data, net, "train")
        val_eval = evaluate_model(data, net, "validation")

        # Step 8: Append the metrics to train_metrics_epoch and val_metrics_epoch
        train_metrics_epoch.append(
            {
                metric: train_eval[metric]
                for metric in params.metrics
                if metric in train_eval
            }
        )
        val_metrics_epoch.append(
            {
                metric: val_eval[metric]
                for metric in params.metrics
                if metric in val_eval
            }
        )

    return {"metrics_val": val_metrics_epoch, "metrics_train": train_metrics_epoch}
    ### END SOLUTION


# TODO: Implement this!
def train_pairwise_spedup(net, params, data):
    """
    This function should train the given network using the sped up pairwise loss


    Note: Do not change the function definition!
    Note: You can assume params.batch_size will always be equal to 1


    net: the neural network to be trained

    params: params is an object which contains config used in training
        (eg. params.epochs - the number of epochs to train).
        For a full list of these params, see the next cell.

    Returns: a dictionary containing: "metrics_val" (a list of dictionaries) and
             "metrics_train" (a list of dictionaries).

             "metrics_val" should contain metrics (the metrics in params.metrics) computed
             after each epoch on the validation set (metrics_train is similar).
             You can use this to debug your models
    """

    val_metrics_epoch = []
    train_metrics_epoch = []
    ### BEGIN SOLUTION
    # Step Tips:
    # Step 1: Create a DataLoader for the training data
    train_data = DataLoader(QueryGroupedLTRData(data, "train"), shuffle=True, collate_fn=qg_collate_fn)

    # Step 2: Create your Adam optimizer
    optimizer = Adam(net.parameters(), lr=params.lr)

    # Step 3: Iterate over the data for the number of epochs
    for epoch in range(params.epochs):
        net.train()

        # Step 4: Iterate over each batch of data and compute the scores using the forward pass of the network
        for batch in train_data:
            features_q_i, labels_q_i = batch
            features_q_i = torch.cat(features_q_i, dim=0)
            labels_q_i = torch.cat(labels_q_i, dim=0)
            
            optimizer.zero_grad()
            scores = net(features_q_i)

            # Step 5: Compute the lambda gradient values for the pairwise loss (spedup) with the compute_lambda_i method on the scores and the output labels
            lambda_i = compute_lambda_i(scores, labels_q_i)

            # Step 6: Backward from the scores with the use of the lambda gradient values
            torch.autograd.backward(scores, lambda_i)

            # Step 7: Update the weights using the optimizer
            optimizer.step()

        # Step 8: At the end of the epoch, evaluate the model on the data using the evaluate_model function (both train and val)
        net.eval()
        eval_train = evaluate_model(data, net, "train")
        eval_val = evaluate_model(data, net, "validation")

        # Step 9: Append the metrics to train_metrics_epoch and val_metrics_epoch
        train_metrics_epoch.append(
            {
                metric: eval_train[metric]
                for metric in params.metrics
                if metric in eval_train
            }
        )
        val_metrics_epoch.append(
            {
                metric: eval_val[metric]
                for metric in params.metrics
                if metric in eval_val
            }
        )

    return {"metrics_val": val_metrics_epoch, "metrics_train": train_metrics_epoch}
    ### END SOLUTION


# TODO: Implement this!
def train_listwise(net, params, data):
    """
    This function should train the given network using the listwise (LambdaRank) loss

    Note: Do not change the function definition!
    Note: You can assume params.batch_size will always be equal to 1


    net: the neural network to be trained

    params: params is an object which contains config used in training
        (eg. params.epochs - the number of epochs to train).
        For a full list of these params, see the next cell.

    Returns: a dictionary containing: "metrics_val" (a list of dictionaries) and
             "metrics_train" (a list of dictionaries).

             "metrics_val" should contain metrics (the metrics in params.metrics) computed
             after each epoch on the validation set (metrics_train is similar).
             You can use this to debug your models
    """

    val_metrics_epoch = []
    train_metrics_epoch = []
    ### BEGIN SOLUTION
    # Step Tips:
    # Step 1: Create a DataLoader for the training data
    train_data = DataLoader(QueryGroupedLTRData(data, "train"), shuffle=True, collate_fn=qg_collate_fn)
    # Step 2: Create your Adam optimizer
    optimizer = Adam(net.parameters(), lr=params.lr)

    # Step 3: Iterate over the data for the number of epochs
    for epoch in range(params.epochs):
        net.train()

        # Step 4: Iterate over each batch of data and compute the scores using the forward pass of the network
        for batch in train_data:
            features_q_i, labels_q_i = batch
            features_q_i = torch.cat(features_q_i, dim=0)
            labels_q_i = torch.cat(labels_q_i, dim=0)
            optimizer.zero_grad()
            scores = net(features_q_i)

            # Step 5: Compute the lambda gradient values for the listwise (LambdaRank) loss with the compute_lambda_i method on the scores and the output labels
            lambda_i = listwise_loss(scores, labels_q_i)

            # Step 6: Bacward from the scores with the use of the lambda gradient values
            torch.autograd.backward(scores, lambda_i)

            # Step 7: Update the weights using the optimizer
            optimizer.step()

        # Step 8: At the end of the epoch, evaluate the model on the data using the evaluate_model function (both train and val)
        net.eval()
        eval_train = evaluate_model(data, net, "train")
        eval_val = evaluate_model(data, net, "validation")

        # Step 9: Append the metrics to train_metrics_epoch and val_metrics_epoch
        train_metrics_epoch.append(
            {
                metric: eval_train[metric]
                for metric in params.metrics
                if metric in eval_train
            }
        )
        val_metrics_epoch.append(
            {
                metric: eval_val[metric]
                for metric in params.metrics
                if metric in eval_val
            }
        )

    return {"metrics_val": val_metrics_epoch, "metrics_train": train_metrics_epoch}
    ### END SOLUTION
