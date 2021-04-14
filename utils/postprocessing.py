import abc
from PIL import Image
import tempfile
import numpy as np
from skimage.filters import gaussian


class Postprocessing(abc.ABC):
    @abc.abstractmethod
    def __call__(self, img):
        """
        Apply post-processing
        :param img: ndarray with float32 intensity values in range [0, 255]
        :return: ndarray with float32 intensity values in range [0, 255]
        """
        pass

    @classmethod
    @abc.abstractmethod
    def name(cls):
        pass

    @abc.abstractmethod
    def details(self):
        """
        Provide post-processing filter details
        :return: dict with post-processing parameters as key-value pairs
        """
        pass


class IdentityPostprocessing(Postprocessing):
    def __call__(self, img):
        return img

    @classmethod
    def name(cls):
        return "Identity"

    def details(self):
        return {}


class JpegPostprocessing(Postprocessing):
    def __init__(self, quality_factor):
        super().__init__()
        self.quality_factor = quality_factor

    @classmethod
    def name(cls):
        return "Jpeg"

    def __call__(self, img):
        """
        Apply JPEG compression
        :param img: image with pixel intensities in range [0, 255], can be float32 or uint8 dtype
        :return: img with intensities in range [0, 255], float32 dtype
        """

        # Convert to uint8
        img_uint8 = np.clip(img, 0, 255).round().astype(np.uint8)

        # Apply JPEG compression
        with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
            im = Image.fromarray(img_uint8)
            im.save(f.name, quality=int(self.quality_factor))
            # Read back in
            im_recovered = Image.open(f.name)
            img_recovered = np.array(im_recovered)

        return img_recovered.astype(np.float32)

    def details(self):
        return {
            "quality_factor": self.quality_factor
        }


class AdditiveNoiseSNRPostprocessing(Postprocessing):
    def __init__(self, noise_SNR_db):
        super().__init__()
        self.noise_SNR_db = noise_SNR_db

    @classmethod
    def name(cls):
        return "AdditiveNoiseSNR"

    def __call__(self, img):
        noise_SNR = np.power(10, self.noise_SNR_db / 10)

        img_variance = np.var(img)

        # SNR is currently assumed to be var(img) / var(noise)
        # We can also use SNR = 10 * log_{10} (var(img) / var(noise))
        noise_variance = img_variance / noise_SNR

        if len(img.shape) == 2:
            img = img + np.sqrt(noise_variance) * np.random.randn(img.shape[0], img.shape[1])
        elif len(img.shape) == 3 and img.shape[2] == 3:
            img[:, :, :3] = img + np.sqrt(noise_variance) * np.random.randn(img.shape[0], img.shape[1], 3)
        else:
            raise RuntimeError("Unexpected input size")

        return img

    def details(self):
        return {
            "noise_SNR_db": self.noise_SNR_db
        }


class GaussianBlurPostprocessing(Postprocessing):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    @classmethod
    def name(cls):
        return "GaussianBlur"

    def __call__(self, img):
        img_blurred = gaussian(img, sigma=self.sigma, preserve_range=True)
        return img_blurred

    def details(self):
        return {
            "sigma": self.sigma
        }


class AdditiveNoisePostprocessing(Postprocessing):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    @classmethod
    def name(cls):
        return "AdditiveNoise"

    def __call__(self, img):
        img_noisy = img + self.sigma * np.random.randn(*img.shape)
        img_noisy = np.clip(img_noisy, 0, 255)
        return img_noisy

    def details(self):
        return {
            "sigma": self.sigma
        }
