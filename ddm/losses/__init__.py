"""The loss terms of the objective.

    style      L_MM  moments matching, L_CM  correlation matching  (Style Matching module)
    diversity  L_ICD intra-class diversity                          (ICD module)

The content term L_MMD is the plain distribution-matching objective and is computed inline
in ddm.engine.condense, since it is a single line and has no variants.
"""
from ddm.losses.diversity import (content_diversity_loss, intra_class_diversity_loss,
                                  style_diversity_loss)
from ddm.losses.style import (calc_mean_std, correlation_matching_loss, gram_matrix,
                              moments_matching_loss, style_vector)

__all__ = ['calc_mean_std', 'gram_matrix', 'style_vector', 'moments_matching_loss',
           'correlation_matching_loss', 'style_diversity_loss', 'content_diversity_loss',
           'intra_class_diversity_loss']
