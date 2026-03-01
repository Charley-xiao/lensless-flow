import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import numpy as np
from torch.utils.data import Dataset
from abc import abstractmethod
import cv2
import torchvision.transforms as tf
from waveprop.noise import add_shot_noise
import os
import warnings
# from lensless.utils.dataset import DiffuserCamMirflickrHF

torch_available = True
RPI_HQ_CAMERA_CCM_MATRIX = np.array(
    [
        [2.0659, -0.93119, -0.13421],
        [-0.11615, 1.5593, -0.44314],
        [0.073694, -0.4368, 1.3636],
    ]
)
RPI_HQ_CAMERA_BLACK_LEVEL = 256.3
SUPPORTED_BIT_DEPTH = np.array([8, 10, 12, 16])
FLOAT_DTYPES = [np.float32, np.float64]

def get_max_val(img, nbits=None):
    """
    For uint image.

    Parameters
    ----------
    img : :py:class:`~numpy.ndarray`
        Image array.
    nbits : int, optional
        Number of bits per pixel. Detect if not provided.

    Returns
    -------
    max_val : int
        Maximum pixel value.
    """
    assert img.dtype not in FLOAT_DTYPES
    if nbits is None:
        nbits = int(np.ceil(np.log2(img.max())))

    if nbits not in SUPPORTED_BIT_DEPTH:
        nbits = SUPPORTED_BIT_DEPTH[nbits < SUPPORTED_BIT_DEPTH][0]
    max_val = 2**nbits - 1
    if img.max() > max_val:
        new_nbit = int(np.ceil(np.log2(img.max())))
        print(f"Detected pixel value larger than {nbits}-bit range, using {new_nbit}-bit range.")
        max_val = 2**new_nbit - 1
    return max_val

def bayer2rgb_cc(
    img,
    nbits,
    down=None,
    blue_gain=None,
    red_gain=None,
    black_level=RPI_HQ_CAMERA_BLACK_LEVEL,
    ccm=RPI_HQ_CAMERA_CCM_MATRIX,
    nbits_out=None,
):
    """
    Convert raw Bayer data to RGB with the following steps:

    #. Demosaic with bi-linear interpolation, mapping the Bayer array to RGB.
    #. Black level removal.
    #. White balancing, applying gains to red and blue channels.
    #. Color correction matrix.
    #. Clip.

    Parameters
    ----------
    img : :py:class:`~numpy.ndarray`
        2D Bayer data to convert to RGB.
    nbits : int
        Bit depth of input data.
    blue_gain : float
        Blue gain.
    red_gain : float
        Red gain.
    black_level : float
        Black level. Default is to use that of Raspberry Pi HQ camera.
    ccm : :py:class:`~numpy.ndarray`
        Color correction matrix. Default is to use that of Raspberry Pi HQ camera.
    nbits_out : int
        Output bit depth. Default is to use that of input.

    Returns
    -------
    rgb : :py:class:`~numpy.ndarray`
        RGB data.
    """
    assert len(img.shape) == 2, img.shape
    if nbits_out is None:
        nbits_out = nbits
    if nbits_out > 8:
        dtype = np.uint16
    else:
        dtype = np.uint8

    # demosaic Bayer data
    img = cv2.cvtColor(img, cv2.COLOR_BayerRG2RGB)

    # downsample
    if down is not None:
        img = resize(img[None, ...], factor=1 / down, interpolation=cv2.INTER_CUBIC)[0]

    # correction
    img = img - black_level
    if red_gain:
        img[:, :, 0] *= red_gain
    if blue_gain:
        img[:, :, 2] *= blue_gain
    img = img / (2**nbits - 1 - black_level)
    img[img > 1] = 1

    img = (img.reshape(-1, 3, order="F") @ ccm.T).reshape(img.shape, order="F")
    img[img < 0] = 0
    img[img > 1] = 1
    return (img * (2**nbits_out - 1)).astype(dtype)


def print_image_info(img):
    """
    Print dimensions, data type, max, min, mean.
    """
    print("dimensions : {}".format(img.shape))
    print("data type : {}".format(img.dtype))
    print("max  : {}".format(img.max()))
    print("min  : {}".format(img.min()))
    print("mean : {}".format(img.mean()))


def load_image(
    fp,
    verbose=False,
    flip=False,
    flip_ud=False,
    flip_lr=False,
    bayer=False,
    black_level=RPI_HQ_CAMERA_BLACK_LEVEL,
    blue_gain=None,
    red_gain=None,
    ccm=RPI_HQ_CAMERA_CCM_MATRIX,
    back=None,
    nbits_out=None,
    as_4d=False,
    downsample=None,
    bg=None,
    return_float=False,
    shape=None,
    dtype=None,
    normalize=True,
    bgr_input=True,
):
    """
    Load image as numpy array.

    Parameters
    ----------
    fp : str
        Full path to file.
    verbose : bool, optional
        Whether to plot into about file.
    flip : bool
        Whether to flip data (vertical and horizontal).
    bayer : bool
        Whether input data is Bayer.
    blue_gain : float
        Blue gain for color correction.
    red_gain : float
        Red gain for color correction.
    black_level : float
        Black level. Default is to use that of Raspberry Pi HQ camera.
    ccm : :py:class:`~numpy.ndarray`
        Color correction matrix. Default is to use that of Raspberry Pi HQ camera.
    back : array_like
        Background level to subtract.
    nbits_out : int
        Output bit depth. Default is to use that of input.
    as_4d : bool
        Add depth and color dimensions if necessary so that image is 4D: (depth,
        height, width, color).
    downsample : int, optional
        Downsampling factor. Recommended for image reconstruction.
    bg : array_like
        Background level to subtract.
    return_float : bool
        Whether to return image as float array, or unsigned int.
    shape : tuple, optional
        Shape (H, W, C) to resize to.
    dtype : str, optional
        Data type of returned data. Default is to use that of input.
    normalize : bool, default True
        If ``return_float``, whether to normalize data to maximum value of 1.

    Returns
    -------
    img : :py:class:`~numpy.ndarray`
        RGB image of dimension (height, width, 3).
    """
    assert os.path.isfile(fp)

    nbits = None  # input bit depth
    if "dng" in fp:
        import rawpy

        assert bayer
        raw = rawpy.imread(fp)
        img = raw.raw_image
        # # # TODO : use raw.postprocess? to much unknown processing...
        # img = raw.postprocess(
        #     adjust_maximum_thr=0,  # default 0.75
        #     no_auto_scale=False,
        #     # no_auto_scale=True,
        #     gamma=(1, 1),
        #     bright=1,  # default 1
        #     exp_shift=1,
        #     no_auto_bright=True,
        #     # use_camera_wb=True,
        #     # use_auto_wb=False,
        #     # -- gives better balance for PSF measurement
        #     use_camera_wb=False,
        #     use_auto_wb=True,  # default is False? f both use_camera_wb and use_auto_wb are True, then use_auto_wb has priority.
        # )

        # if red_gain is None or blue_gain is None:
        #     camera_wb = raw.camera_whitebalance
        #     red_gain = camera_wb[0]
        #     blue_gain = camera_wb[1]

        nbits = int(np.ceil(np.log2(raw.white_level)))
        ccm = raw.color_matrix[:, :3]
        black_level = np.array(raw.black_level_per_channel[:3]).astype(np.float32)
    elif "npy" in fp or "npz" in fp:
        img = np.load(fp)
    else:
        img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)

    if bayer:
        assert len(img.shape) == 2, img.shape
        if nbits is None:
            if img.max() > 255:
                # HQ camera
                nbits = 12
            else:
                nbits = 8

        if back:
            back_img = cv2.imread(back, cv2.IMREAD_UNCHANGED)
            dtype = img.dtype
            img = img.astype(np.float32) - back_img.astype(np.float32)
            img = np.clip(img, a_min=0, a_max=img.max())
            img = img.astype(dtype)
        if nbits_out is None:
            nbits_out = nbits

        img = bayer2rgb_cc(
            img,
            nbits=nbits,
            blue_gain=blue_gain,
            red_gain=red_gain,
            black_level=black_level,
            ccm=ccm,
            nbits_out=nbits_out,
        )

    else:
        if len(img.shape) == 3 and bgr_input:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    original_dtype = img.dtype

    if flip:
        img = np.flipud(img)
        img = np.fliplr(img)
    if flip_ud:
        img = np.flipud(img)
    if flip_lr:
        img = np.fliplr(img)

    if bg is not None:

        # if bg is float vector, turn into int-valued vector
        if bg.max() <= 1 and img.dtype not in [np.float32, np.float64]:
            bg = bg * get_max_val(img)

        img = img - bg
        img = np.clip(img, a_min=0, a_max=img.max())

    if as_4d:
        if len(img.shape) == 3:
            img = img[np.newaxis, :, :, :]
        elif len(img.shape) == 2:
            img = img[np.newaxis, :, :, np.newaxis]

    if downsample is not None or shape is not None:
        if downsample is not None:
            factor = 1 / downsample
        else:
            factor = None
        img = resize(img, factor=factor, shape=shape)

    if return_float:
        if dtype is None:
            dtype = np.float32
        assert dtype == np.float32 or dtype == np.float64
        img = img.astype(dtype)
        if normalize:
            img /= img.max()

    else:
        if dtype is None:
            dtype = original_dtype
        img = img.astype(dtype)

    if verbose:
        print_image_info(img)

    return img

def resize(img, factor=None, shape=None, interpolation=cv2.INTER_CUBIC):
    """
    Resize by given factor.

    Parameters
    ----------
    img : :py:class:`~numpy.ndarray`
        Image to downsample
    factor : int or float
        Resizing factor.
    shape : tuple
        Shape to copy ([depth,] height, width, color). If provided, (height, width) is used.
    interpolation : OpenCV interpolation method
        See https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#cv2.resize

    Returns
    -------
    img : :py:class:`~numpy.ndarray`
        Resized image.
    """
    min_val = img.min()
    max_val = img.max()
    img_shape = np.array(img.shape)[-3:-1]

    assert not ((factor is None) and (shape is None)), "Must specify either factor or shape"
    new_shape = tuple(img_shape * factor) if shape is None else shape[-3:-1]
    new_shape = [int(i) for i in new_shape]

    if np.array_equal(img_shape, new_shape):
        return img

    if torch_available:
        # torch resize expects an input of form [color, depth, width, height]
        tmp = np.moveaxis(img, -1, 0)
        tmp = torch.from_numpy(tmp.copy())
        resized = tf.Resize(size=new_shape, antialias=True)(tmp).numpy()
        resized = np.moveaxis(resized, 0, -1)

    else:
        resized = np.array(
            [
                cv2.resize(img[i], dsize=tuple(new_shape[::-1]), interpolation=interpolation)
                for i in range(img.shape[-4])
            ]
        )
        # OpenCV discards channel dimension if it is 1, put it back
        if len(resized.shape) == 3:
            # resized = resized[:, :, :, np.newaxis]
            resized = np.expand_dims(resized, axis=-1)

    return np.clip(resized, min_val, max_val)

class DualDataset(Dataset):
    """
    Abstract class for defining a dataset of paired lensed and lensless images.
    """

    def __init__(
        self,
        indices=None,
        # psf_path=None,
        background=None,
        # background_pix=(0, 15),
        downsample=1,
        flip=False,
        flip_ud=False,
        flip_lr=False,
        transform_lensless=None,
        transform_lensed=None,
        input_snr=None,
        **kwargs,
    ):
        """
        Dataset consisting of lensless and corresponding lensed image.

        Parameters
        ----------
        indices : range or int or None
            Indices of the images to use in the dataset (if integer, it should be interpreted as range(indices)), by default None.
        psf_path : str
            Path to the PSF of the imaging system, by default None.
        background : :py:class:`~torch.Tensor` or None, optional
            If not ``None``, background is removed from lensless images, by default ``None``. If PSF is provided, background is estimated from the PSF.
        background_pix : tuple, optional
            Pixels to use for background estimation, by default (0, 15).
        downsample : int, optional
            Downsample factor of the lensless images, by default 1.
        flip : bool, optional
            If ``True``, lensless images are flipped, by default ``False``.
        transform_lensless : PyTorch Transform or None, optional
            Transform to apply to the lensless images, by default ``None``. Note that this transform is applied on HWC images (different from torchvision).
        transform_lensed : PyTorch Transform or None, optional
            Transform to apply to the lensed images, by default ``None``. Note that this transform is applied on HWC images (different from torchvision).
        input_snr : float, optional
            If not ``None``, Poisson noise is added to the lensless images to match the given SNR.
        """
        if isinstance(indices, int):
            indices = range(indices)
        self.indices = indices
        self.background = background
        self.input_snr = input_snr
        self.downsample = downsample
        self.flip = flip
        self.flip_ud = flip_ud
        self.flip_lr = flip_lr
        self.transform_lensless = transform_lensless
        self.transform_lensed = transform_lensed

        # self.psf = None
        # if psf_path is not None:
        #     psf, background = load_psf(
        #         psf_path,
        #         downsample=downsample,
        #         return_float=True,
        #         return_bg=True,
        #         bg_pix=background_pix,
        #     )
        #     if self.background is None:
        #         self.background = background
        #     self.psf = torch.from_numpy(psf)
        #     if self.transform_lensless is not None:
        #         self.psf = self.transform_lensless(self.psf)

    @abstractmethod
    def __len__(self):
        """
        Abstract method to get the length of the dataset. It should take into account the indices parameter.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_images_pair(self, idx):
        """
        Abstract method to get the lensed and lensless images. Should return a pair (lensless, lensed) of numpy arrays with values in [0,1].

        Parameters
        ----------
        idx : int
            images index
        """
        raise NotImplementedError

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        if self.indices is not None:
            idx = self.indices[idx]
        lensless, lensed = self._get_images_pair(idx)

        if isinstance(lensless, np.ndarray):
            # expected case
            if self.downsample != 1.0:
                lensless = resize(lensless, factor=1 / self.downsample)
                lensed = resize(lensed, factor=1 / self.downsample)

            lensless = torch.from_numpy(lensless)
            lensed = torch.from_numpy(lensed)
        else:
            # torch tensor
            # This mean get_images_pair returned a torch tensor. This isn't recommended, if possible get_images_pair should return a numpy array
            # In this case it should also have applied the downsampling
            pass

        # If [H, W, C] -> [D, H, W, C]
        if len(lensless.shape) == 3:
            lensless = lensless.unsqueeze(0)
        if len(lensed.shape) == 3:
            lensed = lensed.unsqueeze(0)

        if self.background is not None:
            lensless = lensless - self.background
            lensless = torch.clamp(lensless, min=0)

        # add noise
        if self.input_snr is not None:
            lensless = add_shot_noise(lensless, self.input_snr)

        # flip image x and y if needed
        if self.flip:
            lensless = torch.rot90(lensless, dims=(-3, -2), k=2)
            lensed = torch.rot90(lensed, dims=(-3, -2), k=2)
        if self.flip_ud:
            lensless = torch.flip(lensless, dims=(-4, -3))
            lensed = torch.flip(lensed, dims=(-4, -3))
        if self.flip_lr:
            lensless = torch.flip(lensless, dims=(-4, -2))
            lensed = torch.flip(lensed, dims=(-4, -2))
        if self.transform_lensless:
            lensless = self.transform_lensless(lensless)
        if self.transform_lensed:
            lensed = self.transform_lensed(lensed)

        return lensless, lensed
    
def load_psf(
    fp,
    downsample=1,
    return_float=True,
    bg_pix=(5, 25),
    return_bg=False,
    flip=False,
    flip_ud=False,
    flip_lr=False,
    verbose=False,
    bayer=False,
    blue_gain=None,
    red_gain=None,
    dtype=np.float32,
    nbits_out=None,
    single_psf=False,
    shape=None,
    use_3d=False,
    bgr_input=True,
    force_rgb=False,
):
    """
    Load and process PSF for analysis or for reconstruction.

    Basic steps are:
    * Load image.
    * (Optionally) subtract background. Recommended.
    * (Optionally) resize to more manageable size
    * (Optionally) normalize within [0, 1] if using for reconstruction; otherwise cast back to uint for analysis.

    Parameters
    ----------
    fp : str
        Full path to file.
    downsample : int, optional
        Downsampling factor. Recommended for image reconstruction.
    return_float : bool, optional
        Whether to return PSF as float array, or unsigned int.
    bg_pix : tuple, optional
        Section of pixels to take from top left corner to remove background level. Set to `None` to omit this
        step, althrough it is highly recommended.
    return_bg : bool, optional
        Whether to return background level, for removing from data for reconstruction.
    flip : bool, optional
        Whether to flip up-down and left-right.
    verbose : bool
        Whether to print metadata.
    bayer : bool
        Whether input data is Bayer.
    blue_gain : float
        Blue gain for color correction.
    red_gain : float
        Red gain for color correction.
    dtype : float32 or float64
        Data type of returned data.
    nbits_out : int
        Output bit depth. Default is to use that of input.
    single_psf : bool
        Whether to sum RGB channels into single PSF, same across channels. Done
        in "Learned reconstructions for practical mask-based lensless imaging"
        of Kristina Monakhova et. al.

    Returns
    -------
    psf : :py:class:`~numpy.ndarray`
        4-D array of PSF.
    """

    # load image data and extract necessary channels
    if use_3d:
        assert os.path.isfile(fp)
        if fp.endswith(".npy"):
            psf = np.load(fp)
        elif fp.endswith(".npz"):
            archive = np.load(fp)
            if len(archive.files) > 1:
                print("Warning: more than one array in .npz archive, using first")
            elif len(archive.files) == 0:
                raise ValueError("No arrays in .npz archive")
            psf = np.load(fp)[archive.files[0]]
        else:
            raise ValueError("File format not supported")
    else:
        psf = load_image(
            fp,
            verbose=False,
            flip=flip,
            flip_ud=flip_ud,
            flip_lr=flip_lr,
            bayer=bayer,
            blue_gain=blue_gain,
            red_gain=red_gain,
            nbits_out=nbits_out,
            bgr_input=bgr_input,
        )

    original_dtype = psf.dtype
    max_val = get_max_val(psf)
    psf = np.array(psf, dtype=dtype)

    if force_rgb:
        if len(psf.shape) == 2:
            psf = np.stack([psf] * 3, axis=2)
        elif len(psf.shape) == 3:
            pass

    if use_3d:
        if len(psf.shape) == 3:
            grayscale = True
            psf = psf[:, :, :, np.newaxis]
        else:
            assert len(psf.shape) == 4
            grayscale = False

    else:
        if len(psf.shape) == 3:
            grayscale = False
            psf = psf[np.newaxis, :, :, :]
        else:
            assert len(psf.shape) == 2
            grayscale = True
            psf = psf[np.newaxis, :, :, np.newaxis]

    # check that all depths of the psf have the same shape.
    for i in range(len(psf)):
        assert psf[0].shape == psf[i].shape

    # subtract background, assume black edges
    if bg_pix is None:
        bg = np.zeros(len(np.shape(psf)))

    else:
        # grayscale
        if grayscale:
            bg = np.mean(psf[:, bg_pix[0] : bg_pix[1], bg_pix[0] : bg_pix[1], :])
            psf -= bg

        # rgb
        else:
            bg = []
            for i in range(psf.shape[3]):
                bg_i = np.mean(psf[:, bg_pix[0] : bg_pix[1], bg_pix[0] : bg_pix[1], i])
                psf[:, :, :, i] -= bg_i
                bg.append(bg_i)

        psf = np.clip(psf, a_min=0, a_max=psf.max())
        bg = np.array(bg)

    # resize
    if downsample != 1 or shape is not None:
        psf = resize(psf, shape=shape, factor=1 / downsample)

    if single_psf:
        if not grayscale:
            # TODO : in Lensless Learning, they sum channels --> `psf_diffuser = np.sum(psf_diffuser,2)`
            # https://github.com/Waller-Lab/LenslessLearning/blob/master/pre-trained%20reconstructions.ipynb
            psf = np.sum(psf, axis=3)
            psf = psf[:, :, :, np.newaxis]
        else:
            warnings.warn("Notice : single_psf has no effect for grayscale psf")
            single_psf = False

    # normalize
    if return_float:
        # psf /= psf.max()
        psf /= np.linalg.norm(psf.ravel())
        bg /= max_val
    else:
        psf = psf.astype(original_dtype)

    if verbose:
        print_image_info(psf)

    if return_bg:
        return psf, bg
    else:
        return psf

class DiffuserCamMirflickrHF(DualDataset):
    def __init__(
        self,
        split,
        repo_id="bezzam/DiffuserCam-Lensless-Mirflickr-Dataset",
        psf="psf.tiff",
        downsample=2,
        flip_ud=True,
        dtype="float32",
        **kwargs,
    ):
        """
        Parameters
        ----------
        split : str
            Split of the dataset to use: 'train', 'test', or 'all'.
        downsample : int, optional
            Downsample factor of the PSF, which is 4x the resolution of the images, by default 6 for resolution of (180, 320).
        flip_ud : bool, optional
            If True, data is flipped up-down, by default ``True``. Otherwise data is upside-down.
        """

        # get dataset
        self.dataset = load_dataset(repo_id, split=split)

        # get PSF
        psf_fp = hf_hub_download(repo_id=repo_id, filename=psf, repo_type="dataset")
        psf, bg = load_psf(
            psf_fp,
            verbose=False,
            downsample=downsample * 4,
            return_bg=True,
            flip_ud=flip_ud,
            dtype=dtype,
            bg_pix=(0, 15),
        )
        self.psf = torch.from_numpy(psf)

        super(DiffuserCamMirflickrHF, self).__init__(
            flip_ud=flip_ud, downsample=downsample, background=bg, **kwargs
        )

    def __len__(self):
        return len(self.dataset)

    def _get_images_pair(self, idx):
        lensless = np.array(self.dataset[idx]["lensless"])
        lensed = np.array(self.dataset[idx]["lensed"])

        # normalize
        lensless = lensless.astype(np.float32) / 255
        lensed = lensed.astype(np.float32) / 255

        return lensless, lensed


def make_dataloader(split: str, downsample: int, flip_ud: bool, batch_size: int, num_workers: int, path: str = "bezzam/DiffuserCam-Lensless-Mirflickr-Dataset"):
    ds = DiffuserCamMirflickrHF(split=split, downsample=downsample, flip_ud=flip_ud, repo_id=path)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"),
                    num_workers=num_workers, pin_memory=True, drop_last=(split == "train"), 
                    persistent_workers=(num_workers > 0), prefetch_factor=4 if num_workers > 0 else None)
    return ds, dl