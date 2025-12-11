def pgd_attack(model, images, labels, epsilon, alpha, iters):

    ori_images = images.clone().detach()
    # model.eval()  # optionnel, cf remarque

    for _ in range(iters):
        images = images.clone().detach().requires_grad_(True)

        with torch.enable_grad():
            outputs = model(images)
            loss = criterion(outputs, labels)

        model.zero_grad(set_to_none=True)
        loss.backward()

        grad_sign = images.grad.sign()
        adv_images = images + alpha * grad_sign

        eta = torch.clamp(adv_images - ori_images, min=-epsilon, max=epsilon)
        images = torch.clamp(ori_images + eta, 0, 1)

    return images.detach()
