def display_images(image, model, epsilon):
    _, label, confidence = get_imagenet_label(model.predict(image))
    plt.figure()
    plt.imshow(image[0]*0.5+0.5)
    plt.title(
        f"""
    Epsilon = {epsilon:0.3f} \n 
    {label} : confiance de {100 * confidence:.2f}%
    """
    )
    plt.show()
