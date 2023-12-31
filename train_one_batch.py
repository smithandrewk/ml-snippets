lossi = []

model.train()
for i in range(1):
    for (X_tr,y_tr) in tqdm(trainloader):
        X_tr,y_tr = X_tr.to(DEVICE),y_tr.to(DEVICE)
        logits = model(X_tr)
        loss = criterion(logits,y_tr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lossi.append(loss.item())
plt.plot(torch.tensor(lossi[:len(lossi) - len(lossi)%10]).view(-1,10).mean(axis=1))
print(torch.tensor(lossi[:len(lossi) - len(lossi)%10]).view(-1,10).mean(axis=1)[-1])
