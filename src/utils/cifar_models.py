from peewee import (
    SqliteDatabase,
    Model,
    AutoField,
    IntegerField,
    FloatField,
    TextField,
    DateTimeField,
    ForeignKeyField,
)
from datetime import datetime


db = SqliteDatabase("oe.db", autoconnect=False)


class BaseModel(Model):
    class Meta:
        database = db


class CIFARModel(BaseModel):
    id = AutoField()
    description = TextField(null=True)
    latent_dim_size = IntegerField()
    outliers_num = IntegerField()
    normal_data_num = IntegerField()
    lambda_1 = FloatField()
    lambda_2 = FloatField()
    gen_lr = FloatField()
    disc_lr = FloatField()
    gen_lr_milestones = TextField()
    disc_lr_milestones = TextField()
    created_at = DateTimeField(default=datetime.now)
    interpolations_sample_size = IntegerField()
    normal_label = IntegerField(null=True)
    outliers_label = IntegerField(null=True)
    outliers_labels = TextField(null=True)
    normal_labels = TextField(null=True)
    batch_size = IntegerField()
    weights_path = TextField(null=True)
    means = TextField(null=True)
    stds = TextField(null=True)
    lambda_3 = FloatField(null=True)


class CIFARScore(BaseModel):
    id = AutoField()
    model = ForeignKeyField(CIFARModel, backref="scores")
    epoch = IntegerField()
    seed = IntegerField()
    score = FloatField()


class TrainingResult(BaseModel):
    model = ForeignKeyField(CIFARModel, backref="trainingresults")
    epoch = IntegerField()
    seed = IntegerField()
    g_loss = FloatField()
    d_loss = FloatField(null=True)
    g_val_loss = FloatField(null=True)
    d_val_loss = FloatField(null=True)
    weights_path = TextField()
    image_path = TextField()
    created_at = DateTimeField(default=datetime.now)


class Experiment(BaseModel):
    id = AutoField()

    model = ForeignKeyField(CIFARModel, backref="experiments")
    training_epoch = ForeignKeyField(TrainingResult, backref="experiments", null=True)
    normal_labels = TextField(null=False)
    outliers_labels = TextField(null=False)
    outliers_num = TextField(null=False)
    auc_mean = FloatField(null=False)
    f1_mean = FloatField(null=False)
    average_prediction_score_mean = FloatField(null=False)
    auc_std = FloatField(null=False)
    f1_std = FloatField(null=False)
    average_prediction_score_std = FloatField(null=False)
    num_seeds = IntegerField(null=False)
    created_at = DateTimeField(default=datetime.now)


class TestResult(BaseModel):
    model = ForeignKeyField(CIFARModel, backref="testresults")
    test_label = IntegerField(null=True)
    average_prediction_score = FloatField()
    auc = FloatField()
    f1_score = FloatField(null=True)
    seed = IntegerField(null=True)
    outliers_num = IntegerField(null=True)
    f1_scores = TextField(null=True)
    # experiment = ForeignKeyField('Experiment', backref="testresults", null=True)


class EvaluationResult(BaseModel):
    id = AutoField()

    model = ForeignKeyField(CIFARModel, backref="evaluations")
    epoch = IntegerField()
    f1_score = FloatField()
    auc = FloatField()
    average_prediction_score = FloatField()


class EvalResult(BaseModel):
    id = AutoField()

    normal_labels = TextField(null=False)
    model_ids = TextField(null=False)
    outliers_labels = TextField(null=False)
    model_outliers_num = IntegerField(null=False)
    test_outliers_num = IntegerField(null=False)
    f1_scores = TextField(null=False)
    f1_score = TextField(null=False)
    auc_score = TextField(null=False)
    average_prediction_score = TextField(null=False)
    created_at = DateTimeField(default=datetime.now)
    auprc_scores = TextField(null=False)
    test_label = IntegerField(null=False)


def init_db():
    with db:
        db.create_tables(
            [
                CIFARModel,
                TrainingResult,
                TestResult,
                CIFARScore,
                EvaluationResult,
                Experiment,
                EvalResult,
            ]
        )
