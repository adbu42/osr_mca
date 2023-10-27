import matplotlib.pyplot as plt


def show_image(image_tensor):
    plt.figure()
    plt.imshow(image_tensor.permute(1, 2, 0))
    plt.show()
