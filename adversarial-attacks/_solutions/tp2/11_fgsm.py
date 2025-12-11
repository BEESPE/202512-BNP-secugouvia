epsilons = [0, 0.01, 0.1, 0.15, 0.3]  # par exemple

for i, eps in enumerate(epsilons):
    adv_x = image + eps*perturbations
    adv_x = tf.clip_by_value(adv_x, -1, 1)
    display_images(adv_x, pretrained_model, eps)
