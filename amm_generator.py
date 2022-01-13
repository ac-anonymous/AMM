import numpy
from scipy.sparse import csr_matrix

class AmmGenerator:

    def greedy_combined_selection_itv(self, x_dataset_matrix: csr_matrix, s_matrix_shap: numpy.ndarray, feature_names: list, trigger_size=75):

        if type(x_dataset_matrix) is csr_matrix:
            x_local = x_dataset_matrix.toarray()
        else:
            x_local = x_dataset_matrix

        shap_matrix_t = s_matrix_shap.T
        distances_feature = numpy.array([item.max() - item.min() for item in shap_matrix_t])

        feature_count = shap_matrix_t.shape[0]
        sample_count = x_local.shape[0]

        d_p = []

        for index in range(0, feature_count):
            distance = distances_feature[index]
            shap_vec = shap_matrix_t[index]
            mean = numpy.mean(shap_vec)
            p = numpy.sum(shap_vec > mean) / sample_count  # find vi > mean
            d_p.append(distance * p)

        order_feature = numpy.argsort(-numpy.array(d_p))
        output = {}
        for _index in range(0, trigger_size):
            order = order_feature[_index]
            feature = feature_names[order]
            shap_vec = shap_matrix_t[order]
            row = numpy.argmin(shap_vec)
            value = x_local[row][order]
            output[feature] = value

        return output