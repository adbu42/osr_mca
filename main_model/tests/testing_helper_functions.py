import matplotlib.pyplot as plt


def show_image(image_tensor, image_dataset):
    image_tensor = image_dataset.reverse_normalization(image_tensor)
    plt.figure()
    plt.imshow(image_tensor.permute(1, 2, 0))
    plt.show()
