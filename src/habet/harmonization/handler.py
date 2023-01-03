import pandas as pd
import numpy as np
import pickle
import itk

from pathlib import Path
from ..util import ITKImageMetadata, image_from_array
from . import registry

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

    def construct_data_matrix(self, mask=None):
        image_data_matrix = []
        ref_meta = None
        num_voxels = None
        xs, ys, zs = None, None, None
        num_samples = self.df.shape[0]

        # Read in mask
        if mask is not None:
            ref_meta = ITKImageMetadata(mask)
            mask_arr = itk.array_from_image(mask)
            xs, ys, zs = np.nonzero(mask_arr)

        for _, row in self.df.iterrows():
            image = itk.imread(row["image_path"])
            image_meta = ITKImageMetadata(image)
            image_arr = itk.array_from_image(image)

            if ref_meta is None:
                ref_meta = image_meta
            else:
                assert image_meta == ref_meta

            if mask is None:
                voxel_values = image_arr.flatten()
            else:
                voxel_values = image_arr[xs, ys, zs]

            image_data_matrix.append(voxel_values)
            if num_voxels is None:
                num_voxels = voxel_values.size
            else:
                assert voxel_values.size == num_voxels

        image_data_matrix = np.array(image_data_matrix).T
        assert image_data_matrix.shape == (num_voxels, num_samples)
        return image_data_matrix, ref_meta

    def handle(self):
        registry_dict = registry.get_registry_dict()
        mask = None
        if self.mask_path is not None:
            mask = itk.imread(str(self.mask_path))

        data_matrix, data_meta = self.construct_data_matrix(mask=mask)
        for method_name in self.harmonization_methods_to_use:
            output_dir_for_method = self.output_dir / method_name
            image_dir_for_method = output_dir_for_method / "harmonized_images"
            output_dir_for_method.mkdir(exist_ok=True)
            image_dir_for_method.mkdir(exist_ok=True)

            method_class = registry_dict[method_name]
            method_instance = method_class(
                data_matrix, self.df, self.site_colname, covariate_cols=self.covariate_cols
            )
            ret = method_instance.harmonize()

            with open(str(output_dir_for_method / "harmonize_info_dict.p"), "wb") as fp:
                pickle.dump(ret, fp)


            harmonized_im_to_site_id = self.df[[self.site_colname, "image_path"]].copy()
            harmonized_im_to_site_id["image_path"] = harmonized_im_to_site_id["image_path"].apply(lambda x: str((image_dir_for_method / Path(x).name).resolve()))

            harmonized_im_to_site_id.to_csv(output_dir_for_method / "image_to_site.csv", index=False)
            for i, im_flat in enumerate(ret["data"].T):
                out_path = harmonized_im_to_site_id.iloc[i, :]["image_path"]

                # Note that we can assume that image_i
                # corresponds to the ith row in the df because thats how it
                # was set up
                if mask is not None:
                    xs, ys, zs = np.nonzero(itk.array_from_image(mask))

                    original_im_path = self.df.iloc[i, :]["image_path"]
                    im = itk.array_from_image(itk.imread(original_im_path))
                    im[xs, ys, zs] = im_flat
                    im = image_from_array(im, data_meta)
                else:
                    im = image_from_array(im_flat, data_meta)

                itk.imwrite(im, out_path)
