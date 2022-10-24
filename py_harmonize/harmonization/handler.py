import pandas as pd
import numpy as np
import dipy.io.image
import pickle

from monai.transforms import LoadImage
from pathlib import Path
from .base import _registry


class HarmonizationHandler:
    def __init__(
        self,
        output_dir,
        harmonization_methods_to_use,
        df_path,
        site_colname,
        covariate_cols=None,
        mask_path=None,
    ):
        self.output_dir = output_dir
        self.harmonization_methods_to_use = harmonization_methods_to_use
        self.df_path = df_path
        self.df = pd.read_csv(self.df_path)
        self.site_colname = site_colname
        self.covariate_cols = covariate_cols
        self.mask_path = mask_path

        self.image_loader = LoadImage(image_only=False)

    def construct_data_matrix(self, mask=None, mask_affine=None):
        # Prepare data for harmonization
        if (mask is None and mask_affine is not None) or (
            mask is not None and mask_affine is None
        ):
            raise ValueError("Mask and mask_affine must both be specified or None")

        image_data_matrix = []
        affine = None
        num_voxels = None
        xs, ys, zs = None, None, None
        num_samples = self.df.shape[0]
        im_shapes = None

        # Read in mask
        if mask is not None:
            affine = mask_affine
            xs, ys, zs = np.nonzero(mask)

        for _, row in self.df.iterrows():
            image, image_header = self.image_loader(row["image_path"])
            image = np.array(image)

            if affine is None:
                affine = image_header["affine"]
            else:
                np.testing.assert_array_almost_equal(image_header["affine"], affine)

            if im_shapes is None:
                im_shapes = image.shape
            else:
                assert image.shape == im_shapes

            if mask is None:
                voxel_values = image.flatten()
            else:
                voxel_values = image[xs, ys, zs]

            image_data_matrix.append(voxel_values)
            if num_voxels is None:
                num_voxels = voxel_values.size
            else:
                voxel_values.size == num_voxels

        image_data_matrix = np.array(image_data_matrix).T
        assert image_data_matrix.shape == (num_voxels, num_samples)
        return image_data_matrix, im_shapes, affine

    # TODO: Probably a fast numpy way of doing this
    def _reshape_harmonized_data(self, harmonized_voxels, shape):
        stack = []
        for flat_im in harmonized_voxels.T:
            stack.append(np.reshape(flat_im, shape))

        return stack

    # TODO: Might want to combine this with step where we write images back out
    # would save us another iteration over the dataframe and some memory
    def _paste_voxels_back_on_original_ims(self, harmonized_voxels, mask):
        xs, ys, zs = np.nonzero(mask)
        stack = []
        for i, (_, row) in enumerate(self.df.iterrows()):
            image, _ = self.image_loader(row["image_path"])
            image = np.array(image)
            image[xs, ys, zs] = harmonized_voxels[:, i]
            stack.append(image)

        return stack

    def handle(self):
        mask = None
        mask_affine = None
        if self.mask_path is not None:
            mask, mask_meta = self.image_loader(self.mask_path)
            mask_affine = np.array(mask_meta["affine"])
            mask = np.array(mask)

        data_matrix, original_im_shapes, affine = self.construct_data_matrix(
            mask=mask, mask_affine=mask_affine
        )
        for method_name in self.harmonization_methods_to_use:
            output_dir_for_method = self.output_dir / method_name
            image_dir_for_method = output_dir_for_method / "harmonized_images"
            output_dir_for_method.mkdir(exist_ok=True)
            image_dir_for_method.mkdir(exist_ok=True)

            method_class = _registry[method_name]
            method_instance = method_class(
                data_matrix, self.df, self.site_colname, covariate_cols=self.covariate_cols
            )
            ret = method_instance.harmonize()

            with open(str(output_dir_for_method / "harmonize_info_dict.p"), "wb") as fp:
                pickle.dump(ret, fp)

            harmonized_im_stack = None
            if mask is None:
                harmonized_im_stack = self._reshape_harmonized_data(
                    ret["data"], original_im_shapes
                )
            else:
                harmonized_im_stack = self._paste_voxels_back_on_original_ims(
                    ret["data"], mask
                )

            harmonized_im_to_site_id = self.df[[self.site_colname, "image_path"]].copy()
            harmonized_im_to_site_id["image_path"] = harmonized_im_to_site_id["image_path"].apply(lambda x: str((image_dir_for_method / Path(x).name).resolve()))

            harmonized_im_to_site_id.to_csv(output_dir_for_method / "image_to_site.csv", index=False)
            for i, im in enumerate(harmonized_im_stack):
                # Just write back out. Note that we can assume that image_i
                # corresponds to the ith row in the df because thats how it
                # was set up
                out_path = harmonized_im_to_site_id.iloc[i, :]["image_path"]
                dipy.io.image.save_nifti(str(out_path), im, affine)
