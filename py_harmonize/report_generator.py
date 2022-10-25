import pandas as pd
import numpy as np
import pingouin as pg
import dipy.io.image
import pickle
import scipy.stats

from monai.transforms import LoadImage
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

class ReportGenerator:
    def __init__(
        self,
        output_dir,
        im_path_to_site_id_df_path,
        site_colname,
        anova_alpha,
        t_test_alpha,
        mtc,
        save_dfs,
        mask_path=None,
    ):
        self.output_dir = output_dir
        self.im_path_to_site_id_df_path = im_path_to_site_id_df_path
        self.site_colname = site_colname
        self.anova_alpha = anova_alpha
        self.t_test_alpha = t_test_alpha
        self.mtc = mtc
        self.save_dfs = save_dfs
        self.mask_path = mask_path

        self.image_loader = LoadImage(image_only=False)
        self.im_path_to_site_id_df = pd.read_csv(str(self.im_path_to_site_id_df_path))
        if self.mask_path is None:
            self.mask, self.mask_affine = None, None
        else:
            self.mask, self.mask_affine = self.image_loader(str(mask_path))

    def _generate_voxel_wise_df(self):
        master_df = pd.DataFrame(
            columns=[self.site_colname, "x", "y", "z", "voxel_value"]
        )

        # All images should have the same header / metadata info
        master_affine = None

        xs, ys, zs = None, None, None
        if self.mask is not None:
            xs, ys, zs = np.nonzero(self.mask)

        # Note that we go in order
        for i, (_, row) in enumerate(self.im_path_to_site_id_df.iterrows()):
            img_data, img_meta = self.image_loader(row["image_path"])
            if i == 0:
                master_affine = img_meta["affine"]
            else:
                assert (img_meta["affine"] == master_affine).all()

            if xs is None and ys is None and zs is None:
                temp_mask = np.ones(img_data.shape)
                xs, ys, zs = np.nonzero(temp_mask)

            voxel_values = img_data[xs, ys, zs]

            df_for_image = pd.DataFrame(
                {
                    self.site_colname: row[self.site_colname],
                    "x": xs,
                    "y": ys,
                    "z": zs,
                    "voxel_value": voxel_values,
                }
            )
            master_df = pd.concat([master_df, df_for_image], ignore_index=True)

        return master_df, master_affine

    def _generate_stats_dfs(self, voxel_wise_df):
        # Read in dataframe, set up some other variables, etc.
        groups = voxel_wise_df.groupby(["x", "y", "z"])
        num_voxels = groups.ngroups
        anova_alpha = self.anova_alpha / num_voxels if self.mtc else self.anova_alpha
        t_test_padjust = "bonf" if self.mtc else "none"
        num_significant_anovas = 0
        t_tests_master_df = pd.DataFrame(
            columns=[
                "x",
                "y",
                "z",
                "Contrast",
                "A",
                "B",
                "Paired",
                "Parametric",
                "T",
                "dof",
                "alternative",
                "p-unc",
                "p-corr",
                "p-adjust",
                "BF10",
                "hedges",
            ]
        )
        anova_master_df = pd.DataFrame()

        # Main work
        for (x, y, z), group in groups:
            # First handle anova
            anova_for_voxel = pg.anova(data=group, dv="voxel_value", between=self.site_colname)
            anova_for_voxel.insert(0, "x", x)
            anova_for_voxel.insert(1, "y", y)
            anova_for_voxel.insert(2, "z", z)
            anova_master_df = pd.concat(
                [anova_master_df, anova_for_voxel], ignore_index=True
            )

            # Note that, despite the use of p-unc here, we're actually doing
            # a bonferroni correction, since ANOVA_ALPHA may be corrected
            if anova_for_voxel["p-unc"].iat[0] <= anova_alpha:
                num_significant_anovas += 1

                # Now calculate t-tests
                t_tests_for_voxel = pg.pairwise_tests(
                    dv="voxel_value",
                    between=self.site_colname,
                    data=group,
                    padjust=t_test_padjust,
                )
                t_tests_for_voxel.insert(0, "x", x)
                t_tests_for_voxel.insert(1, "y", y)
                t_tests_for_voxel.insert(2, "z", z)
                t_tests_master_df = pd.concat(
                    [t_tests_master_df, t_tests_for_voxel], ignore_index=True
                )

        anova_master_df["significant"] = anova_master_df["p-unc"] <= anova_alpha

        # If MTC is True, we wont know the t test alpha value until we've done
        # all of the anovas
        if t_tests_master_df.shape[0]:
            T_TEST_ALPHA = (
                self.t_test_alpha / num_significant_anovas
                if self.mtc
                else self.t_test_alpha
            )
            t_tests_master_df["significant"] = (
                t_tests_master_df["p-corr"] <= T_TEST_ALPHA
            )
        else:
            t_tests_master_df["significant"] = []

        t_tests_master_df.x = t_tests_master_df.x.astype("int64")
        t_tests_master_df.y = t_tests_master_df.y.astype("int64")
        t_tests_master_df.z = t_tests_master_df.z.astype("int64")

        return anova_master_df, t_tests_master_df

    def voxel_df_to_arr(self, xs, ys, zs, intensity_col, image_shape=None):
        if image_shape is None:
            image_shape = (xs.max() + 1, ys.max() + 1, zs.max() + 1)

        # Add one here because of 0-based indexing
        ret = np.zeros(image_shape)

        ret[xs, ys, zs] = intensity_col

        return ret

    def save_with_pickle(self, obj, path):
        with open(str(path), "wb") as fp:
            pickle.dump(obj, fp)

    def _generate_from_stats(self, anova_df, t_test_df):
        # We'll get the max x y and z from the anova table,
        # since that has one entry for every x y and z value.
        # If we didn't do this and only looked at the t-tests, the resulting
        # image might be smaller, since there are only rows for voxels with
        # significant f-statistics
        max_x = anova_df["x"].max()
        max_y = anova_df["y"].max()
        max_z = anova_df["z"].max()
        image_shape = (max_x + 1, max_y + 1, max_z + 1)

        ##### Anova images #####
        anova_df["masked_np2"] = anova_df["np2"] * anova_df["significant"]
        np2_im = self.voxel_df_to_arr(
            anova_df.x, anova_df.y, anova_df.z, anova_df.masked_np2
        )
        sig_locs_im = self.voxel_df_to_arr(
            anova_df.x,
            anova_df.y,
            anova_df.z,
            anova_df.significant.astype(np.uint8),
            image_shape=image_shape,
        )

        ##### t-tests images#####
        def compute_t_test_stats_per_voxel_group(g):
            num_t_tests_for_voxel = g.shape[0]
            num_sig = g["significant"].sum()
            return pd.Series(
                {"frac_sig": num_sig / num_t_tests_for_voxel, "num_sig": num_sig}
            )

        # If there are voxels to examine, compute features
        if t_test_df.shape[0]:
            voxel_wise_t_test_info = t_test_df.groupby(
                ["x", "y", "z"], as_index=False
            ).apply(compute_t_test_stats_per_voxel_group)
        # Otherwise just make an empty dataframe with the correct columns
        else:
            voxel_wise_t_test_info = pd.DataFrame(
                columns=["x", "y", "z", "frac_sig", "num_sig"]
            )
        sig_t_test_im = self.voxel_df_to_arr(
            voxel_wise_t_test_info.x,
            voxel_wise_t_test_info.y,
            voxel_wise_t_test_info.z,
            voxel_wise_t_test_info.frac_sig,
            image_shape=image_shape,
        )

        ##### Now some final numbers #####
        # First anova
        total_num_anovas = anova_df.shape[0]
        total_num_sig_anovas = anova_df.significant.sum()
        frac_sig_anovas = total_num_sig_anovas / total_num_anovas

        # Now t-tests
        total_num_sig_t_tests = t_test_df.significant.sum()

        # Fisher's method
        _, anova_combined_p_value = scipy.stats.combine_pvalues(anova_df["p-unc"], method="fisher")

        stats_df = pd.DataFrame({
            "total_num_anovas": [total_num_anovas],
            "total_num_sig_anovas": [total_num_sig_anovas],
            "frac_sig_anovas": [frac_sig_anovas],
            "total_num_sig_t_tests": [total_num_sig_t_tests],
            "anova_combined_p_value": [anova_combined_p_value]

        })

        assert stats_df.shape[0] == 1

        return (
            np2_im,
            sig_locs_im,
            sig_t_test_im,
            stats_df,
        )

    def generate_report(self):
        self.output_dir.mkdir(exist_ok=True, parents=True)
        voxel_wise_df, affine = self._generate_voxel_wise_df()
        anova_df, t_test_df = self._generate_stats_dfs(voxel_wise_df)
        (
            np2_im,
            sig_locs_im,
            sig_t_test_im,
            aggregate_stats_df,
        ) = self._generate_from_stats(anova_df, t_test_df)

        ##### IO #####
        # Images
        dipy.io.image.save_nifti(str(self.output_dir / "anova_np2.nii.gz"), np2_im, affine)
        dipy.io.image.save_nifti(str(self.output_dir / "anova_significance.nii.gz"), sig_locs_im, affine)
        dipy.io.image.save_nifti(str(self.output_dir / "t_test_frac_significance.nii.gz"), sig_t_test_im, affine)

        # Numbers
        aggregate_stats_df.to_csv(str(self.output_dir / "stats.csv"), index=False)

        # dfs
        if self.save_dfs:
            anova_df.to_csv(self.output_dir / "anova_df.csv", index=False)
            t_test_df.to_csv(self.output_dir / "t_test_df.csv", index=False)
