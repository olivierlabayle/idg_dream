from sklearn.pipeline import Pipeline

from idg_dream.transformers import InchiLoader, SequenceLoader, ProteinEncoder

baseline_pipeline = Pipeline(steps=[('load_inchis', InchiLoader),
                                    ('load_sequences', SequenceLoader),
                                    ('encode_proteins', ProteinEncoder),
                                    ('encode_ecfp')])