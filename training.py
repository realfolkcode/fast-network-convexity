import torch


def loss_dist(X, A, idx, norm=1):
    dist = torch.cdist(X, X, p=norm)
    k = dist.shape[0] * len(idx) - len(idx)
    s = torch.sum((dist[idx] - A[idx])**2) / k
    return s


def loss_log_dist(X, A, idx, norm=1):
    dist = torch.cdist(X, X, p=norm)
    k = dist.shape[0] * len(idx) - len(idx)
    s = torch.sum((torch.log(dist[idx] + 1e-9) - torch.log(A[idx] + 1e-9))**2) / k
    return s


def l1_loss_dist(X, A, idx, norm=1):
    dist = torch.cdist(X, X, p=norm)
    k = dist.shape[0] * len(idx) - len(idx)
    s = torch.sum(torch.abs(dist[idx] - A[idx])) / k
    return s


def train(model, num_epochs, g, dist, loader, opt, scheduler, log_loss=False, max_grad_norm=2., verbose=1):
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for idx in loader:
            emb = model(g, g.ndata['x'])
            if log_loss:
                loss = loss_log_dist(emb, dist, idx)
            else:
                loss = loss_dist(emb, dist, idx)
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
        emb = model(g, g.ndata['x'])
    loss = loss_dist(emb, dist, g.nodes())
    print(f'Final loss: {loss.item()}')
    loss = loss_log_dist(emb, dist, g.nodes())
    print(f'Final loss (log): {loss.item()}')
    J = l1_loss_dist(emb, dist, g.nodes())
    print(f'Absolute loss (J): {J}')

    return emb