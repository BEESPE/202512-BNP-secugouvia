adv_images_cw = cw_l22_attack(model, images, labels, c=1e-3, iters=50)
outputs_cw = model(adv_images_cw)

print("Accuracy CW:", (outputs_cw.argmax(1) == labels).float().mean().item())
