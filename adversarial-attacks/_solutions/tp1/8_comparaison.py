epsilon = 0.1  # par exemple

dataiter = iter(testloader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)


adv_images_fgsm = fgsm_attack(model, images, labels, epsilon=0.1)
adv_images_pgd = pgd_attack(model, images, labels,
                            epsilon, alpha=0.01, iters=10)

outputs_orig = model(images)
outputs_fgsm = model(adv_images_fgsm)
outputs_pgd = model(adv_images_pgd)

print("Accuracy original:", (outputs_orig.argmax(1) == labels).float().mean().item())
print("Accuracy FGSM:", (outputs_fgsm.argmax(1) == labels).float().mean().item())
print("Accuracy PGD:", (outputs_pgd.argmax(1) == labels).float().mean().item())
