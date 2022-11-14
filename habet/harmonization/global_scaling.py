import numpy as np

from .base import HarmonizationMethod


class GlobalScaling(HarmonizationMethod):
    def __init__(self, data_matrix, df, site_colname, covariate_cols=None):
        super().__init__(data_matrix, df, site_colname, covariate_cols)

    def _harmonize(self):
        # TODO: Some data type checking
        sorted_labels, site_ids = np.unique(self.df[self.site_colname], return_inverse=True)
        num_sites = len(sorted_labels)

        num_voxels = self.data_matrix.shape[0]

        # First get the average voxel intensities across all of the scans
        y_bar = np.mean(self.data_matrix, axis=1)
        y_bar = np.expand_dims(y_bar, 1)
        assert y_bar.shape == (num_voxels, 1)

        X = np.concatenate([np.ones((num_voxels, 1)), y_bar], axis=1)
        assert X.shape == (num_voxels, 2)

        Z = []
        # Now construct the Z matrix
        for i in range(num_sites):
            indices_scans_at_site_i = site_ids == i
            y_i = np.mean(self.data_matrix[:, indices_scans_at_site_i], axis=1)
            Z.append(y_i)

        Z = np.array(Z).T
        assert Z.shape == (num_voxels, num_sites)

        # Now the easy part, use the normal equation to get the
        # location and scale params.
        beta_hat = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Z)

        harmonized_data = []

        # Harmonize the images
        for im_i, site_id in zip(self.data_matrix.T, site_ids):
            theta_loc_i, theta_scale_i = beta_hat[:, site_id]

            harmonized_im_i = (im_i - theta_loc_i) / theta_scale_i
            harmonized_data.append(harmonized_im_i)

        harmonized_data = np.array(harmonized_data).T

        out_dict = {"data": harmonized_data, "estimates": {"beta_hat": beta_hat}}
        return out_dict
