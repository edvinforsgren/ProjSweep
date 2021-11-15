"""
AZUREML logger
------
"""
import os
from argparse import Namespace
from time import time
from typing import Optional, Dict, Any, Union

try:
    import azureml.core
    from azureml.core import Experiment, Workspace, run
    _AZUREML_AVAILABLE = True
except ImportError:  # pragma: no-cover
    _AZUREML_AVAILABLE = False

from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment

class AzureLogger(LightningLoggerBase):
    """
    Log using `AzureML <https://studio.azureml.net/>`_. Install it with pip:
    .. code-block:: bash
        pip install azureml-core
    Example:
        >>> from pytorch_lightning import Trainer
        >>> from dl_imaging_kit.logger import AzureLogger
        >>> azure_logger = AzureLogger(
        ...     experiment_name="default",
        ...     tracking_uri="file:./ml-runs"
        ... )
        >>> trainer = Trainer(logger=azure_logger)
    Use the logger anywhere in you :class:`~pytorch_lightning.core.lightning.LightningModule` as follows:
    >>> from pytorch_lightning import LightningModule
    >>> class LitModel(LightningModule):
    ...     def training_step(self, batch, batch_idx):
    ...         # example
    ...         self.logger.experiment.whatever_azure_log_supports(...)
    ...
    ...     def any_lightning_module_function_or_hook(self):
    ...         self.logger.experiment.whatever_azure_log_supports(...)
    Args:
        experiment_name: The name of the experiment
        azure_config: "The path to the config file or starting directory to search for azure workspace config file"
            if not provided, will start the search from os.chmod()
        tags: A dictionary tags for the experiment.
    """

    def __init__(self,
                 experiment_name: str = 'default',
                 azure_config: Optional[str] = None,
                 tags: Optional[Dict[str, Any]] = None,
                 save_dir: Optional[str] = None):

        if not _AZUREML_AVAILABLE:
            raise ImportError('You want to use `azure` logger which is not installed yet,'
                              ' install it with `pip install azureml-core`.')
        super().__init__()

        self._run_id = None

        assert experiment_name is not None, "Experiment name is needed"

        print(f" Azure version: {azureml.core.__version__}")
        if azure_config is None:
            ws = Workspace.from_config()
        else:
            ws = Workspace.from_config(azure_config)
        print("Initiating an azure logger ",
              'Workspace name: ' + ws.name,
              'Azure region: ' + ws.location,
              'Subscription id: ' + ws.subscription_id,
              'Resource group: ' + ws.resource_group, sep='\n')

        self.workspace = ws
        self.experiment_name = experiment_name
        self._experiment = None

    @property
    @rank_zero_experiment
    def experiment(self) -> run:
        r"""
        Actual AzureMl run object. To use AzureMl features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.
        Example::
            self.logger.experiment.some_azureml_function()

        https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.run(class)?view=azure-ml-py
        """
        if self._experiment is not None:
            return self._experiment

        self.exp = Experiment(workspace=self.workspace, name=self.experiment_name)
        self._experiment = self.exp.start_logging(snapshot_directory=None)

        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params: Dict[str, Any]) -> None:

        self.experiment.add_properties(params)

    @rank_zero_only
    def log_tags(self, params: Dict[str, Any]) -> None:

        for k, v in params.items():
            self.experiment.tag(k, v)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        assert rank_zero_only.rank == 0, 'experiment tried to log from global_rank != 0'

        for k, v in metrics.items():
            self.experiment.log(k, v)

    @rank_zero_only
    def finalize(self, status: str = "success") -> None:
        """
        :param status: str - (e.g. success, failed, aborted)
        """
        super().finalize(status)
        if status != 'success':
            self.experiment.fail(status)
        #self.experiment.complete()

    @rank_zero_only
    def finish(self, status: str = "success") -> None:
    
        super().finalize(status)
        if status != 'success':
            self.experiment.fail(status)
        self.experiment.complete()

    @rank_zero_only
    def log_image(self, name: str, image) -> None:
        """
        :param name: name of the image
        :param image: matplotlib or path to an image
        """
        self.experiment.log_image(name, image)

    @rank_zero_only
    def log_file(self, name: str, path_to_file: str) -> None:
        self.experiment.upload_file(name, path_to_file)

    @property
    def name(self) -> str:
        return self.experiment_name

    @property
    def version(self) -> str:

        return self.experiment.id