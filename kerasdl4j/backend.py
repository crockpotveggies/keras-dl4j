import h5py
import tempfile
import new
import numpy
import xxhash

from keras import backend as K
from os import path, mkdir
from py4j.java_gateway import JavaGateway

batch_file_template = "batch_{id}.h5"
hijack_cache = {}


def generate_tmp_path():
    tmp_file = tempfile.NamedTemporaryFile(prefix="dl4j")
    tmp_file.close()

    return tmp_file.name


def dump_h5(dataset, batch_size, directory_name):
    """
    Dumps the data from dataset to a series of HDF5 files. Each of them will contain at most batch_size samples.

    :param dataset: Dataset to store
    :param batch_size: Size of the batch
    :param directory_name: Directory where the batch files are going to be saved
    """

    if path.exists(directory_name):
        raise IOError("Path exists: " + directory_name)
        return

    mkdir(directory_name)

    batch_id = 0
    samples_count = dataset.shape[0]

    begin = 0
    end = batch_size

    while begin < samples_count:
        batch_file_name = batch_file_template.format(id=batch_id)
        f = h5py.File(path.join(directory_name, batch_file_name), 'w')
        f.create_dataset("data", data=dataset[begin:end])
        f.flush()
        f.close()

        begin = end
        end += batch_size
        batch_id += 1


def hash_ndarray(array):
    """
    Calculates a hash of contents of ndarray
    :param array: Array to calculate hash
    :return: hex digest of the hash (as string)
    """

    hsh = xxhash.xxh64()
    hsh.update(array.view(numpy.uint8))
    return hsh.hexdigest() 


def dump_ndarray(batch_size, dataset):
    dataset_hash = hash_ndarray(dataset)
    if not dataset_hash in hijack_cache:
        directory_name = generate_tmp_path()
        dump_h5(dataset, batch_size, directory_name)
        hijack_cache[dataset_hash] = directory_name
    else:
        print("Dataset already dumped")

    return hijack_cache[dataset_hash]


def check_dl4j_model(
    model_object):
    """
    Checks the current Keras model object in scope
    and installs a reference to DL4J MultiLayerNetwork
    if it doesn't exist.
    """
    if hasattr(model_object, '_dl4j_model'):
        return model_object
    else:
        model_file_path = generate_tmp_path()
        model_object.save(model_file_path)

        gateway = JavaGateway()
        modelType = None

        if model.__class__.__name__ == 'Sequential':
            modelType = gateway.jvm.org.deeplearning4j.keras.model.KerasModelType.SEQUENTIAL
            params_builder = gateway.jvm.org.deeplearning4j.keras.api.KerasModelRef.builder()
            params_builder.type(modelType)
            params_builder.modelFilePath(model_file_path)

            model_object._dl4j_model = gateway.sequential_to_multilayernetwork(params_builder.build())
            model_object._dl4j_type = modelType

        elif model.__class__.__name__ == 'Model':
            modelType = gateway.jvm.org.deeplearning4j.keras.model.KerasModelType.FUNCTIONAL
            params_builder = gateway.jvm.org.deeplearning4j.keras.api.KerasModelRef.builder()
            params_builder.type(modelType)
            params_builder.modelFilePath(model_file_path)

            model_object._dl4j_model = gateway.functional_to_computationgraph(params_builder.build())
            model_object._dl4j_type = modelType
        else:
            raise ValueError('DL4J Keras only works with Sequential and Functional models')
        
        return model_object


def install_dl4j_backend(model):
    """
    Hijacks the `fit` method call in the model object. Detects
    if model is Sequential or Functional.
    :param model: Model in which fit will be hijacked
    """
    # append special methods
    # save_model()
    model.save_model = new.instancemethod(_save_model, model, None)

    # hijack Keras API
    if model.__class__.__name__ == 'Sequential':
        # compile()
        model._old_compile = model.compile
        model.compile = new.instancemethod(_sequential_compile)
        # fit()
        model._old_fit = model.fit
        model.fit = new.instancemethod(_sequential_fit, model, None)
        # evaluate()
        model._old_evaluate = model.evaluate
        model.evaluate = new.instancemethod(_sequential_evaluate, model, None)
        # predict()
        model._old_predict = model.predict
        model.predict = new.instancemethod(_sequential_predict, model, None)
        # predict_on_batch()
        model._old_predict_on_batch = model.predict_on_batch
        model.predict_on_batch = new.instancemethod(_sequential_predict_on_batch, model, None)
        # compile()
        model._old_compile = model.compile
        model.compile = new.instancemethod(_sequential_compile, model, None)

    elif model.__class__.__name__ == 'Model':
        # compile()
        model._old_compile = model.compile
        model.compile = new.instancemethod(_functional_compile)
        # fit()
        model._old_fit = model.fit
        model.fit = new.instancemethod(_functional_fit, model, None)
        # evaluate()
        model._old_evaluate = model.evaluate
        model.evaluate = new.instancemethod(_functional_evaluate, model, None)
        # predict()
        model._old_predict = model.predict
        model.predict = new.instancemethod(_functional_predict, model, None)
        # predict_on_batch()
        model._old_predict_on_batch = model.predict_on_batch
        model.predict_on_batch = new.instancemethod(_functional_predict_on_batch, model, None)
        # compile()
        model._old_compile = model.compile
        model.compile = new.instancemethod(_functional_compile, model, None)

    else:
        raise ValueError('DL4J Keras only works with Sequential and Functional models')


########
# hijacked functions in model class
########

########
# for Sequential(Model) instances
########

def _sequential_compile(
        self, 
        optimizer, 
        loss,
        metrics=None,
        sample_weight_mode=None,
        **kwargs):
    """
    Configures the learning process.
    """
    # first call the old compile() method
    self._old_compile(self, optimizer, loss, metrics, sample_weight_mode, kwargs)

    # then convert to DL4J instance
    check_dl4j_model(self) # enforces dl4j model for model.fn()


def _sequential_fit(
        self,
        x,
        y,
        batch_size=32,
        nb_epoch=10,
        verbose=1,
        callbacks=[],
        validation_split=0.,
        validation_data=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        **kwargs):
    """
    Executes fitting of the model by using DL4J as backend
    :param model: Model to use
    :param nb_epoch: Number of learning epochs
    :param features_directory: Directory with feature batch files
    :param labels_directory: Directory with label batch files
    :return:
    """
    check_dl4j_model(self) # enforces dl4j model for model.fn()

    training_x = None
    training_y = None
    validation_x = None
    validation_y = None
    do_validation = True

    if validation_data:
        training_x = dump_ndarray(batch_size, x)
        training_y = dump_ndarray(batch_size, y)

        if len(validation_data) == 2:
            val_x, val_y = validation_data
            validation_x = dump_ndarray(batch_size, val_x)
            validation_y = dump_ndarray(batch_size, val_y)            
        elif len(validation_data) == 3:
            val_x, val_y, val_sample_weight = validation_data
            validation_x = dump_ndarray(batch_size, val_x)
            validation_y = dump_ndarray(batch_size, val_y) 
        else:
            raise ValueError('Incorrect configuration for validation_data. Must be a tuple of length 2 or 3.')
    elif validation_split and 0. < validation_split < 1.:
        split_at = int(len(x[0]) * (1. - validation_split))
        x, val_x = (slice_X(x, 0, split_at), slice_X(x, split_at))
        y, val_y = (slice_X(y, 0, split_at), slice_X(y, split_at))
        training_x = dump_ndarray(batch_size, x)
        training_y = dump_ndarray(batch_size, y)
        validation_x = dump_ndarray(batch_size, val_x)
        validation_y = dump_ndarray(batch_size, val_y)
    else:
        do_validation = False
        training_x = dump_ndarray(batch_size, x)
        training_y = dump_ndarray(batch_size, y)


    gateway = JavaGateway()

    params_builder = gateway.jvm.org.deeplearning4j.keras.api.FitParams.builder()
    params_builder.sequentialModel(self._dl4j_model)
    params_builder.nbEpoch(nb_epoch)
    params_builder.trainXPath(training_x)
    params_builder.trainYPath(training_y)
    if not validation_x == None:
        params_builder.validationXPath(validation_x)
        params_builder.validationYPath(validation_y)
    params_builder.dimOrdering(K.image_dim_ordering())
    params_builder.doValidation(do_validation)
    gateway.sequentialFit(params_builder.build())


def _sequential_evaluate(
        self, 
        x, 
        y, 
        batch_size=32, 
        verbose=1,
        sample_weight=None, 
        **kwargs):
    """
    Computes the loss on some input data, batch by batch.
    """
    check_dl4j_model(self) # enforces dl4j model for model.fn() 

    features_directory = dump_ndarray(batch_size, x)
    labels_directory = dump_ndarray(batch_size, y)

    gateway = JavaGateway()

    params_builder = gateway.jvm.org.deeplearning4j.keras.api.EvaluateParams.builder()
    params_builder.sequentialModel(self._dl4j_model)
    params_builder.featuresDirectory(features_directory)
    params_builder.labelsDirectory(labels_directory)
    params_builder.batchSize(batch_size)
    gateway.sequentialEvaluate(params_builder.build())


def _sequential_predict(
    self, 
    x, 
    batch_size=32, 
    verbose=0):
    """
    Generates output predictions for the input samples,
    processing the samples in a batched way.
    """
    check_dl4j_model(self) # enforces dl4j model for model.fn()

    features_directory = dump_ndarray(batch_size, x)

    gateway = JavaGateway()

    params_builder = gateway.jvm.org.deeplearning4j.keras.api.PredictParams.builder()
    params_builder.sequentialModel(self._dl4j_model)
    params_builder.featuresDirectory(features_directory)
    params_builder.batchSize(batch_size)
    gateway.sequentialPredict(params_builder.build())
    # TODO


def _sequential_predict_on_batch(
    self, 
    x):
    """
    Returns predictions for a single batch of samples.
    """
    check_dl4j_model(self) # enforces dl4j model for model.fn() 

    features_directory = dump_ndarray(len(x), x)

    gateway = JavaGateway()

    params_builder = gateway.jvm.org.deeplearning4j.keras.api.PredictOnBatchParams.builder()
    params_builder.sequentialModel(self._dl4j_model)
    params_builder.featuresDirectory(features_directory)
    gateway.sequentialPredictOnBatch(params_builder.build())
    # TODO



########
# for Functional(Model) instances
########

def _functional_compile(
        self, 
        optimizer, 
        loss, 
        metrics=None, 
        loss_weights=None,
        sample_weight_mode=None, 
        **kwargs):
    """
    Configures the model for training.
    """
    # first call the old compile() method
    self._old_compile(self, optimizer, loss, metrics, loss_weights, sample_weight_mode, kwargs)

    # then convert to DL4J instance
    check_dl4j_model(self) # enforces dl4j model for model.fn()
    

def _functional_fit(
        self,
        x,
        y,
        batch_size=32,
        nb_epoch=10,
        verbose=1,
        callbacks=[],
        validation_split=0.,
        validation_data=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        **kwargs):
    """
    Executes fitting of the model by using DL4J as backend
    :param model: Model to use
    :param nb_epoch: Number of learning epochs
    :param features_directory: Directory with feature batch files
    :param labels_directory: Directory with label batch files
    :return:
    """
    check_dl4j_model(self) # enforces dl4j model for model.fn()

    training_x = None
    training_y = None
    validation_x = None
    validation_y = None
    do_validation = True

    if validation_data:
        training_x = dump_ndarray(batch_size, x)
        training_y = dump_ndarray(batch_size, y)

        if len(validation_data) == 2:
            val_x, val_y = validation_data
            validation_x = dump_ndarray(batch_size, val_x)
            validation_y = dump_ndarray(batch_size, val_y)            
        elif len(validation_data) == 3:
            val_x, val_y, val_sample_weight = validation_data
            validation_x = dump_ndarray(batch_size, val_x)
            validation_y = dump_ndarray(batch_size, val_y) 
        else:
            raise ValueError('Incorrect configuration for validation_data. Must be a tuple of length 2 or 3.')
    elif validation_split and 0. < validation_split < 1.:
        split_at = int(len(x[0]) * (1. - validation_split))
        x, val_x = (slice_X(x, 0, split_at), slice_X(x, split_at))
        y, val_y = (slice_X(y, 0, split_at), slice_X(y, split_at))
        training_x = dump_ndarray(batch_size, x)
        training_y = dump_ndarray(batch_size, y)
        validation_x = dump_ndarray(batch_size, val_x)
        validation_y = dump_ndarray(batch_size, val_y)
    else:
        do_validation = False
        training_x = dump_ndarray(batch_size, x)
        training_y = dump_ndarray(batch_size, y)


    gateway = JavaGateway()

    params_builder = gateway.jvm.org.deeplearning4j.keras.api.FitParams.builder()
    params_builder.functionalModel(self._dl4j_model)
    params_builder.nbEpoch(nb_epoch)
    params_builder.trainXPath(training_x)
    params_builder.trainYPath(training_y)
    if not validation_x == None:
        params_builder.validationXPath(validation_x)
        params_builder.validationYPath(validation_y)
    params_builder.dimOrdering(K.image_dim_ordering())
    params_builder.doValidation(do_validation)
    gateway.functionalFit(params_builder.build())


def _functional_evaluate(
        self, 
        x, 
        y, 
        batch_size=32, 
        verbose=1,
        sample_weight=None):
    """
    Returns the loss value and metrics values for the model.
    """
    check_dl4j_model(self) # enforces dl4j model for model.fn() 

    features_directory = dump_ndarray(batch_size, x)
    labels_directory = dump_ndarray(batch_size, y)
    # TODO


def _functional_predict(
    self, 
    x, 
    batch_size=32, 
    verbose=0):
    """
    Generates output predictions for the input samples,
    processing the samples in a batched way.
    """
    check_dl4j_model(self) # enforces dl4j model for model.fn()

    features_directory = dump_ndarray(batch_size, x)

    gateway = JavaGateway()

    params_builder = gateway.jvm.org.deeplearning4j.keras.api.PredictParams.builder()
    params_builder.functionalModel(self._dl4j_model)
    params_builder.featuresDirectory(features_directory)
    params_builder.batchSize(batch_size)
    gateway.functionalPredict(params_builder.build())
    # TODO


def _functional_predict_on_batch(
    self, 
    x):
    """
    Returns predictions for a single batch of samples.
    """
    check_dl4j_model(self) # enforces dl4j model for model.fn() 

    features_directory = dump_ndarray(len(x), x)

    gateway = JavaGateway()

    params_builder = gateway.jvm.org.deeplearning4j.keras.api.PredictOnBatchParams.builder()
    params_builder.functionalModel(self._dl4j_model)
    params_builder.featuresDirectory(features_directory)
    gateway.functionalPredictOnBatch(params_builder.build())
    # TODO



########
# for any Model instances
########


def _save_model(
    self, 
    filepath, 
    overwrite=True,
    saveUpdaterState=False):
    """
    Save model to disk in DL4J format.
    """
    check_dl4j_model(self) # enforces dl4j model for model.fn() 

    if model.__class__.__name__ == 'Sequential':
        params_builder = gateway.jvm.org.deeplearning4j.keras.api.SaveParams.builder()
        params_builder.sequentialModel(self._dl4j_model)
        params_builder.writePath(filepath)
        params_builder.saveUpdaterState(saveUpdaterState)
        gateway.sequentialSave(params_builder.build())

    elif model.__class__.__name__ == 'Model':
        params_builder = gateway.jvm.org.deeplearning4j.keras.api.SaveParams.builder()
        params_builder.functionalModel(self._dl4j_model)
        params_builder.writePath(filepath)
        params_builder.saveUpdaterState(saveUpdaterState)
        gateway.functionalSave(params_builder.build())

    else:
        raise ValueError('DL4J Keras only works with Sequential and Functional models')
