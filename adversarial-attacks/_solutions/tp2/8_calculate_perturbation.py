label = get_ohe_label(class_index,  image_probas.shape[-1])
perturbations = create_adversarial_pattern(image, label)
print(perturbations)
