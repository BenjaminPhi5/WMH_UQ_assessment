from pyrobex.robex import robex
import nibabel as nib

def skull_strip(t1_image):
    """
    image: filepath to T1, OR nibabel image object
    """
    if type(t1_image) == str:
        t1_image = nib.load(t1_image)

    stripped, mask = robex(t1_image)
    # mask = nib.nifti1.Nifti1Image(mask.get_fdata(), affine=t1_image.affine, header=t1_image.header)

    return stripped, mask

def skull_strip_and_save(t1_path, out_path, mask_path):
    image = nib.load(t1_path)

    stripped, mask = skull_strip(image)

    nib.save(stripped, out_path)
    nib.save(mask, mask_path)


def apply_mask(image, mask):
    # assmues image and mask are nibabel images.
    image_data = image.get_fdata()
    mask_data = mask.get_fdata()
    new_image = nib.nifti1.Nifti1Image(image_data * mask_data, affine=image.affine, header=image.header)
    return new_image