import torch
from tqdm import tqdm

def drop_lead(tensor, lead_idx):
    if lead_idx is None:
        return tensor
    # drop lead at lead_idx
    else:
        new_tensor = tensor[:, :lead_idx, :]
        new_tensor = torch.cat((new_tensor, tensor[:, lead_idx + 1:, :]), dim=1)
        return new_tensor

def drop_channels(signal, channel_indices):
    if not channel_indices:
        return signal

    channel_indices = sorted(channel_indices, reverse=True)
    for index in channel_indices:
        signal = torch.cat((signal[:, :index], signal[:, index+1:]), dim=1)
    return signal


# def drop_channels(tensor, channel_idxs:list):
#     # If no channel indices provided, return the original tensor
#     if not channel_idxs:
#         return tensor
#
#     # Sort the indices in decreasing order so that the indices of later channels
#     # are not affected by the removal of earlier channels
#     channel_idxs = sorted(channel_idxs, reverse=True)
#
#     # Remove the specified channels one by one
#     new_tensor = tensor
#     for idx in channel_idxs:
#         new_tensor = torch.cat((new_tensor[:, :idx, :], new_tensor[:, idx+1:, :]), dim=1)
#
#     return new_tensor

# def drop_channels(tensor, channel_idxs):
#     """
#     Remove specified channels from the input tensor.
#
#     :param tensor: Input tensor of shape (batch_size, channels, samples)
#     :type tensor: torch.Tensor
#     :param channel_idxs: List of channel indices to remove
#     :type channel_idxs: list
#     :return: Tensor with specified channels removed
#     :rtype: torch.Tensor
#     """
#     if not isinstance(tensor, torch.Tensor):
#         raise ValueError("Input tensor must be a torch.Tensor")
#
#     if tensor.ndim != 3:
#         raise ValueError("Input tensor must have 3 dimensions (batch_size, channels, samples)")
#
#     # If no channel indices provided, return the original tensor
#     if not channel_idxs:
#         return tensor
#
#     # Sort channel indices in descending order
#     sorted_channel_idxs = sorted(channel_idxs, reverse=True)
#
#     # Remove the specified channels
#     for idx in sorted_channel_idxs:
#         if idx < tensor.shape[1]:
#             tensor = torch.cat((tensor[:, :idx, :], tensor[:, idx+1:, :]), dim=1)
#
#     return tensor





def get_real_indexes(channel_idxs):
    # Create a set of all channel indices in the original tensor
    all_channel_idxs = set(range(14))

    # Remove the specified channels one by one and keep track of the actual indices that were removed
    removed_channel_idxs = []
    for idx in channel_idxs:
        actual_idx = list(all_channel_idxs - set(removed_channel_idxs))[idx]
        removed_channel_idxs.append(actual_idx)

    return removed_channel_idxs


def forward_epoch_train(model, dl, loss_function, optimizer, total_loss=0,lead_to_eliminate=[], to_train=False, desc=None,
                  device=torch.device('cpu')):
    with tqdm(total=len(dl), desc=desc, ncols=50) as pbar:
        model = model.to(device)

        y_true_all=[]
        y_pred_all=[]

        for i_batch, (X, y) in enumerate(dl):
            model.train()
            X = X.to(device)
            # Drop list of leads to start with less than 12
            X = drop_channels(X, lead_to_eliminate).to(device)
            y = y.to(device)

            # Forward:
            y_pred = model(X)
            # Loss:
            y_true = torch.squeeze(y.type(torch.float32))
            loss = loss_function(y_pred, y_true)
            total_loss += loss.item()

            # saving y,ypred of current batch
            y_true_all.append(y_true)
            y_pred_all.append(y_pred)
            optimizer.zero_grad()  # zero the gradients to not accumulate their changes.
            loss.backward()  # get gradients
            # Optimization step:
            optimizer.step()
            pbar.update(1)

        # concatenate y to be return as a single tensor
        y_true_all = torch.cat(y_true_all)
        y_pred_all = torch.cat(y_pred_all)

    return total_loss, y_true_all, y_pred_all

def forward_epoch_val(model, dl, loss_function, optimizer, total_loss=0,lead_to_eliminate=[], to_train=False, desc=None,
                  device=torch.device('cpu')):
    with tqdm(total=len(dl), desc=desc, ncols=50) as pbar:
        model = model.to(device)
        y_true_all = [[] for _ in range(13)]
        y_pred_all = [[] for _ in range(13)]
        loss_lst = torch.zeros(13)
        lead_to_eliminate = get_real_indexes(lead_to_eliminate)
        leads_numbers = list(range(12))+[None]
        leads_numbers = [element for element in leads_numbers if element not in lead_to_eliminate]
        for i_batch, (X, y) in enumerate(dl):
            model.eval()
            with torch.inference_mode():
                # Drop list of leads to start with less than 12
                X = drop_channels(X, lead_to_eliminate).to(device)
                # if X.shape[1] != 12:
                #     leads_numbers.remove(None)
                y = y.to(device)
                for i,val in enumerate(leads_numbers):
                    # Drop a single lead to try new combinations
                    X_dropped = drop_lead(X,val).to(device)
                    # Forward:
                    y_pred = model(X_dropped)
                    #Loss:
                    y_true = torch.squeeze(y.type(torch.float32))

                    current_loss = loss_function(y_pred, y_true)
                    loss_lst[i] += current_loss.item()

                    # saving y,ypred of current batch
                    y_true_all[i].append(y_true)
                    y_pred_all[i].append(y_pred)

                    # Progress bar:
                    pbar.update(1)



        y_true_all = [
            torch.cat([inner_tensor for inner_tensor in inner_list], dim=0)
            for inner_list in y_true_all
            if inner_list
        ]

        y_pred_all = [
            torch.cat([inner_tensor for inner_tensor in inner_list], dim=0)
            for inner_list in y_pred_all
            if inner_list
        ]

        y_true_all = torch.stack(y_true_all)
        y_pred_all = torch.stack(y_pred_all)

    # we return lists of y_pred and y_true and tensor of lost
    return loss_lst, y_true_all, y_pred_all



def forward_epoch_test(model, dl, loss_function, optimizer, total_loss=0,lead_to_eliminate=None, to_train=False, desc=None,
                  device=torch.device('cpu')):
    with tqdm(total=len(dl), desc=desc, ncols=50) as pbar:
        model = model.to(device)

        y_true_all=[]
        y_pred_all=[]
        lead_to_eliminate = get_real_indexes(lead_to_eliminate)

        for i_batch, (X, y) in enumerate(dl):
            model.eval()
            with torch.inference_mode():
                X = drop_channels(X, lead_to_eliminate).to(device)
                y = y.to(device)

                # Forward:
                y_pred = model(X)
                #Loss:
                y_true = torch.squeeze(y.type(torch.float32))
                loss = loss_function(y_pred, y_true)
                total_loss += loss.item()

                # saving y,ypred of current batch
                y_true_all.append(y_true)
                y_pred_all.append(y_pred)

                # Progress bar:
                pbar.update(1)

        # concatenate y to be return as a single tensor
        y_true_all = torch.cat(y_true_all)
        y_pred_all = torch.cat(y_pred_all)

    return total_loss, y_true_all, y_pred_all


def forward_epoch_train_UNet(model, dl, loss_function, optimizer, total_loss=0,lead_to_eliminate=[], to_train=False, desc=None,
                  device=torch.device('cpu')):
    with tqdm(total=len(dl), desc=desc, ncols=50) as pbar:
        model = model.to(device)
        for param in model.parameters():
            param.to(device)

        y_true_all=[]
        y_pred_all=[]

        for i_batch, (X, y) in enumerate(dl):
            if to_train:
                model.train()
                X = X.to(device)
                # Drop list of leads to start with less than 12
                X_dropped = drop_channels(X, lead_to_eliminate).to(device)
                # y = y.to(device)

                # Forward:
                X_hat = model(X_dropped)
                # Loss:
                loss = loss_function(X_hat, X)
                total_loss += loss.item()

                y_true_all.append(X)
                y_pred_all.append(X_hat)

                optimizer.zero_grad()  # zero the gradients to not accumulate their changes.
                loss.backward()  # get gradients
                # Optimization step:
                optimizer.step()

                pbar.update(1)
            else:
                model.eval()
                with torch.inference_mode():
                    X = X.to(device)
                    # Drop list of leads to start with less than 12
                    X_dropped = drop_channels(X, lead_to_eliminate).to(device)

                    # Forward:
                    X_hat = model(X_dropped)
                    # Loss:
                    loss = loss_function(X_hat, X)
                    total_loss += loss.item()

                    y_true_all.append(X)
                    y_pred_all.append(X_hat)
                    pbar.update(1)




        # concatenate y to be return as a single tensor
        y_true_all = torch.cat(y_true_all)
        y_pred_all = torch.cat(y_pred_all)

    return total_loss, y_true_all, y_pred_all









# def get_real_indexes(channel_idxs):
#     # Create a list of all channel indices in the original tensor
#     # all_channel_idxs = list(range(orig_shape[1]))
#     all_channel_idxs = list(range(12))
#
#     # Remove the specified channels one by one and keep track of the actual indices that were removed
#     removed_channel_idxs = []
#     for idx in channel_idxs:
#         actual_idx = all_channel_idxs[idx]
#         removed_channel_idxs.append(actual_idx)
#         all_channel_idxs.remove(actual_idx)
#
#     return removed_channel_idxs


# def drop_channels(signal, channel_indexes):
#     """
#     Reduces channels from a signal tensor.
#
#     Args:
#         signal (tensor): The input signal tensor of shape (batch_size, channels, samples)
#         channel_indexes (list): The list of channel indexes to remove
#
#     Returns:
#         tensor: The output signal tensor with reduced channels
#     """
#     # Get the number of channels to keep
#     num_channels_keep = signal.shape[1] - len(channel_indexes)
#
#     # Create an index list of channels to keep
#     channel_keep_idx = [i for i in range(signal.shape[1]) if i not in channel_indexes]
#
#     # Permute the signal to have the channel dimension first
#     signal = signal.permute(1, 0, 2)
#
#     # Get the channels to keep
#     signal_keep = signal[channel_keep_idx]
#
#     # Permute back to the original shape
#     signal_keep = signal_keep.permute(1, 0, 2)
#
#     # Return the reduced tensor
#     return signal_keep[:, :num_channels_keep, :]