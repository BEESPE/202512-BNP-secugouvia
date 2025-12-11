import torchattacks

atk = torchattacks.CW(model, c=1e-3, lr=0.01, steps=500)
adv = atk(images, labels)
outputs = model(adv)
pred = outputs.argmax(1)
accuracy = (pred == labels).float().mean()
print("Accuracy Torchattacks CW:", accuracy)
