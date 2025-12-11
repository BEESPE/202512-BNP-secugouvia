model.eval()
atk = torchattacks.PGD(
    model,
    eps=0.3,
    alpha=0.01,
    steps=40,
    random_start=True,
)
adv_images = atk(images, labels)
outputs = model(adv_images)
pred = outputs.argmax(1)
accuracy = (pred == labels).float().mean().item()

print("Accuracy Torchattacks PGD :", accuracy)
