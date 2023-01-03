import torch
from tqdm import tqdm

# Training loop:
def forward_epoch(model, dl, loss_function, optimizer, total_loss, to_train=False, desc=None,
                  device=torch.device('cpu')):
    with tqdm(total=len(dl), desc=desc, ncols=100) as pbar:
        model = model.to(device)

        y_true_all=[]
        y_pred_all=[]

        for i_batch, (X, y) in enumerate(dl):
            if to_train:
                model.train()
                X = X.to(device)
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
            else:
                model.eval()
                with torch.inference_mode():
                    X = X.to(device)
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

        # concatenate all y to be returned as a single tensor
        y_true_all=torch.cat(y_true_all[:-1])
        y_pred_all=torch.cat(y_pred_all[:-1])

    return total_loss, y_true_all, y_pred_all

