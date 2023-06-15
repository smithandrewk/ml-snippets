lre = torch.linspace(-6,1,len(trainloader))
lrs = 10**lre
lri = []
lossi = []
model.train()
loss_tr_total = 0
for i,(X_tr,ytr) in tqdm(enumerate(trainloader)):
    optimizer = torch.optim.Adam(model.parameters(),lr=lrs[i])
    X_tr,y_tr = X_tr.to(device),y_tr.to(device)
    logits = model(X_tr)
    loss = criterion(logits,y_tr)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_tr_total += loss.item()
    lri.append(lre[i])
    lossi.append(loss.item())
plt.plot(lri,lossi)
