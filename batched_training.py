import torch
import dgl


def loss_dist(X, X_nograd, A, idx, norm=1):
    dist = torch.cdist(X, X_nograd, p=norm)
    k = X_nograd.shape[0] * len(idx) - len(idx)
    s = torch.sum((dist - A[idx])**2) / k
    return s


def loss_log_dist(X, X_nograd, A, idx, norm=1):
    dist = torch.cdist(X, X_nograd, p=norm)
    k = X_nograd.shape[0] * len(idx) - len(idx)
    s = torch.sum((torch.log(dist + 1e-9) - torch.log(A[idx] + 1e-9))**2) / k
    return s


def l1_loss_dist(X, X_nograd, A, idx, norm=1):
    dist = torch.cdist(X, X_nograd, p=norm)
    k = X_nograd.shape[0] * len(idx) - len(idx)
    s = torch.sum(torch.abs(dist - A[idx])) / k
    return s


def train(model, num_epochs, g, dist, loader, opt, scheduler, log_loss=False, max_grad_norm=2., verbose=1):
    model.eval()
    with torch.no_grad():
        mfgs = [dgl.to_block(g) for _ in range(len(model.conv))]
        inputs = mfgs[0].srcdata['x']
        emb_nograd = model(mfgs, inputs)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for input_nodes, output_nodes, mfgs in loader:
            inputs = mfgs[0].srcdata['x']
            emb = model(mfgs, inputs)
            emb_nograd[output_nodes] = emb.detach()
            if log_loss:
                loss = loss_log_dist(emb, emb_nograd, dist, output_nodes)
            else:
                loss = loss_dist(emb, emb_nograd, dist, output_nodes)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            if scheduler is not None:
                scheduler.step()
            epoch_loss += loss.item()
        epoch_loss /= len(loader)
        if epoch % verbose == 0:
            print(f'Epoch: {epoch}, loss: {epoch_loss}')
    
    model.eval()
    with torch.no_grad():
        mfgs = [dgl.to_block(g) for _ in range(len(model.conv))]
        inputs = mfgs[0].srcdata['x']
        emb = model(mfgs, inputs)
    loss = loss_dist(emb, emb, dist, g.nodes())
    print(f'Final loss: {loss.item()}')
    loss = loss_log_dist(emb, emb, dist, g.nodes())
    print(f'Final loss (log): {loss.item()}')
    J = l1_loss_dist(emb, emb, dist, g.nodes())
    print(f'Absolute loss (J): {J}')

    return emb