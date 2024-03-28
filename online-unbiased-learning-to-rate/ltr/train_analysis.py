import torch
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
from .dataset import ClickLTRData, LTRData
from .logging_policy import LoggingPolicy
from .loss import listNet_loss, unbiased_listNet_loss
from .eval import evaluate_model
import numpy as np

# TODO: Implement this!
def logit_to_prob(logit):
    ### BEGIN SOLUTION
    prob = F.sigmoid(logit)
    ### END SOLUTION
    return prob


# TODO: Implement this!
def train_biased_listNet(net, params, data, logging_policy = LoggingPolicy()):
    """
    This function should train the given network using the (biased) listNet loss

    Note: Do not change the function definition!
    Note: You can assume params.batch_size will always be equal to 1


    net: the neural network to be trained

    params: params is an object which contains config used in training
            params.epochs - the number of epochs to train.
            params.lr - learning rate for Adam optimizer.
            params.batch_size - batch size (always equal to 1)

    Returns: a dictionary containing: "metrics_val" (a list of dictionaries).

             "metrics_val" should contain metrics (the metrics in params.metrics) computed
             after each epoch on the validation set.
             You can use this to debug your models
    """
    val_metrics_epoch = []
    assert params.batch_size == 1
    # logging_policy = LoggingPolicy()
    ### BEGIN SOLUTION
    # Step 1: Create the train data loader
    train_loader = DataLoader(ClickLTRData(data, logging_policy), batch_size=params.batch_size, shuffle=True)
    # Step 2: Create the validation data loader
    val_loader = DataLoader(LTRData(data, split='validation'), batch_size=params.batch_size, shuffle=False) # NOTE: What's the point of this when we're not using it?
    # Step 3: Create the Adam optimizer
    optimizer = Adam(net.parameters(), lr=params.lr)
    # Step 4: Iterate over the epochs and data entries
    for epoch in range(params.epochs):
        # Step 5: Train the model using the listNet loss
        net.train()
        for (features, clicks, _) in train_loader:
            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            output = net(features)
            # Compute the loss
            loss = listNet_loss(output, clicks)
            # Backward pass
            loss.backward()
            # Optimize
            optimizer.step()
        
        # Step 6: Evaluate on the validation set every epoch
        net.eval()
        metrics = evaluate_model(data, net, "validation")
        # Step 7: Store the metrics in val_metrics_epoch
        val_metrics_epoch.append(metrics)
    ### END SOLUTION
    return {"metrics_val": val_metrics_epoch}


# TODO: Implement this! 
def train_unbiased_listNet(net, params, data, logging_policy = LoggingPolicy()):
    """
    This function should train the given network using the unbiased_listNet loss

    Note: Do not change the function definition!
    Note: You can assume params.batch_size will always be equal to 1
    Note: For this function, params should also have the propensity attribute


    net: the neural network to be trained

    params: params is an object which contains config used in training
            params.epochs - the number of epochs to train.
            params.lr - learning rate for Adam optimizer.
            params.batch_size - batch size (always equal to 1)
            params.propensity - the propensity values used for IPS in unbiased_listNet

    Returns: a dictionary containing: "metrics_val" (a list of dictionaries).

             "metrics_val" should contain metrics (the metrics in params.metrics) computed
             after each epoch on the validation set.
             You can use this to debug your models
    """
    val_metrics_epoch = []
    assert params.batch_size == 1
    assert hasattr(params, 'propensity')
    # logging_policy = LoggingPolicy()
    ### BEGIN SOLUTION
    clip_low = getattr(params, 'clip_low', 0.01)
    clip_high = getattr(params, 'clip_high', 1)
    # Step 1: Create the train data loader
    train_loader = DataLoader(ClickLTRData(data, logging_policy), batch_size=params.batch_size, shuffle=True)
    # Step 2: Create the validation data loader
    val_loader = DataLoader(LTRData(data, split='validation'), batch_size=params.batch_size, shuffle=False) # NOTE: Same as above
    # Step 3: Create the Adam optimizer
    optimizer = Adam(net.parameters(), lr=params.lr)
    # Step 4: Iterate over the epochs and data entries
    for epoch in range(params.epochs):
        # Step 5: Train the model using the unbiased listNet loss
        net.train()
        for (features, clicks, positions) in train_loader:
            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            output = net(features)
            # Compute the loss
            loss = unbiased_listNet_loss(output, clicks, params.propensity[positions], clip_low = clip_low, clip_high = clip_high)
            # Backward pass
            loss.backward()
            # Optimize
            optimizer.step()
            
        # Step 6: Evaluate on the validation set every epoch
        net.eval()
        metrics = evaluate_model(data, net, "validation")
        # Step 7: Store the metrics in val_metrics_epoch
        val_metrics_epoch.append({metric: val for metric, val in metrics.items() if metric in params.metrics})
    ### END SOLUTION
    return {"metrics_val": val_metrics_epoch}



# TODO: Implement this!
def train_DLA_listNet(net, params, data, logging_policy = LoggingPolicy()):
    """
    This function should simultanously train both of the given networks
    (i.e. net: for relevance estimation, and params.prop_net: for propensity estimation) using the unbiased_listNet loss.

    Note: Do not change the function definition!
    Note: You can assume params.batch_size will always be equal to 1


    net: the neural network to be trained

    params: params is an object which contains config used in training
            params.epochs - the number of epochs to train.
            params.lr - learning rate for relevance parameters.
            params.batch_size - batch size (always equal to 1)
            params.prop_net - the NN used for propensity estimation
            params.prop_lr - learning rate for propensity parameters.

    Returns: a dictionary containing: "metrics_val" (a list of dictionaries) and
             "train_loss_agg" (a list of aggregates over training losses).
             "metrics_val" should contain metrics (the metrics in params.metrics) computed
             after each epoch on the validation set.
             "train_loss_agg" should contain the min, max and mean of training losses.

             You can use the list of estmated propensities to debug your models.
    """
    val_metrics_epoch = []
    estimated_propensities_epoch = []
    assert params.batch_size == 1
    assert hasattr(params, 'prop_net')
    assert hasattr(params, 'prop_lr')
    # logging_policy = LoggingPolicy()
    ### BEGIN SOLUTION
    clip_low = getattr(params, 'clip_low', 0.01)
    clip_high = getattr(params, 'clip_high', 1)

    prop_net = params.prop_net
    # Step 1: Create the train data loader
    train_loader = DataLoader(ClickLTRData(data, logging_policy), batch_size=params.batch_size, shuffle=True)
    # Step 2: Create the validation data loader
    val_loader = DataLoader(LTRData(data, split='validation'), batch_size=params.batch_size, shuffle=False) # NOTE: Same as above
    # Step 3: Create the Adam optimizer
    optimizer = Adam(net.parameters(), lr=params.lr)
    prop_optimizer = Adam(prop_net.parameters(), lr=params.prop_lr)
    # Step 4: Iterate over the epochs and data entries
    train_loss = []
    for epoch in range(params.epochs):
        # Step 5: Train the model using the DLA listNet loss
        net.train()
        prop_net.train()
        for (features, clicks, positions) in train_loader:
            # Zero the gradients
            optimizer.zero_grad()
            prop_optimizer.zero_grad()
            # Forward pass
            output = net(features)
            prop_output = prop_net(positions)
            # Compute the loss
            loss_relevance = unbiased_listNet_loss(output, clicks, logit_to_prob(prop_output.squeeze()), clip_low=clip_low, clip_high=clip_high)
            loss_propensity = unbiased_listNet_loss(prop_output, clicks, logit_to_prob(output.squeeze()), clip_low = clip_low, clip_high = clip_high)
            loss = loss_relevance + loss_propensity
            train_loss.append(loss)
            # Backward pass
            loss.backward()
            # Optimize
            optimizer.step()
            prop_optimizer.step()
        # Step 6: Evaluate on the validation set every epoch
        net.eval()
        prop_net.eval()
        metrics = evaluate_model(data, net, "validation")
        # Step 7: Store the metrics in val_metrics_epoch
        val_metrics_epoch.append({metric: val for metric, val in metrics.items() if metric in params.metrics})
    ### END SOLUTION
    estimated_propensity = logit_to_prob(
        prop_net(torch.arange(logging_policy.topk)).squeeze().data).numpy()
    # print('estimated propensities:', estimated_propensity / estimated_propensity[0])

    # Detach the tensor and convert it to a NumPy array
    train_loss_detached = [x.detach().numpy() for x in train_loss]

    # Use train_loss_detached as a regular NumPy array
    train_loss_agg = [np.min(train_loss_detached), np.max(train_loss_detached), np.mean(train_loss_detached)]

    return {"metrics_val": val_metrics_epoch, 'train_loss_agg':train_loss_agg}
