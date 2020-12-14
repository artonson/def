import io

import PIL.Image
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a jpeg image and
    returns it. The supplied figure is closed and inaccessible after this call."""

    # Save the plot to a jpeg in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg', dpi=300)
    plt.close(figure)
    buf.seek(0)

    # Convert jpeg buffer to torch tensor
    image = PIL.Image.open(buf)
    image = ToTensor()(image)

    return image
