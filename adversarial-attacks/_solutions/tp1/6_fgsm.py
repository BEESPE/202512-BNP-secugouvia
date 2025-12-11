def fgsm_attack(model, images, labels, epsilon):
    images = images.clone().detach().requires_grad_(True)

    outputs = model(images)
    loss = criterion(outputs, labels)

    model.zero_grad(set_to_none=True)
    loss.backward()

    gradient = images.grad
    perturbation = epsilon * gradient.sign()

    adv_images = images + perturbation
    adv_images = torch.clamp(adv_images, 0, 1)

    return adv_images
