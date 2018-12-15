import modules.PretrainedEmbeddings
import modules.masked_cross_entropy
import modules.Vocab
import modules.MovingAvg
import modules.evaluate
import modules.trainer
import modules.ControllableDataset
import modules.ControllableModel
import modules.AttentionDecoder
import modules.ScoreModelDataset
#import modules.evaluator_polarity


# For flake8 compatibility
__all__ = [modules.masked_cross_entropy, modules.PretrainedEmbeddings,  modules.Vocab,
           modules.MovingAvg, modules.evaluate, modules.trainer, modules.ControllableDataset,modules.ControllableModel,
           modules.AttentionDecoder,modules.ScoreModelDataset]
