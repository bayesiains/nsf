from torch import distributions

import utils


class TweakedUniform(distributions.Uniform):
    def log_prob(self, value, context):
        return utils.sum_except_batch(super().log_prob(value))
        # result = super().log_prob(value)
        # if len(result.shape) == 2 and result.shape[1] == 1:
        #     return result.reshape(-1)
        # else:
        #     return result

    def sample(self, num_samples, context):
        return super().sample((num_samples, ))
