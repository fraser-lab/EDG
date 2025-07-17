"""Batch experiment runner for EDG density-guided diffusion experiments.

This module provides functionality to run multiple experiments in batch:
- Multi-GPU parallel execution with model pre-loading
- Queue management for sequential and parallel processing
- Progress tracking and error handling per experiment
- Reuses existing run_experiment() function
- Supports continue-on-error mode with GPU recovery
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import traceback
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from edg.config import BatchExperimentConfig, ExperimentConfig
from edg.experiment_runner import run_experiment
from edg.gpu_model_manager import GPUModelManager

logger = logging.getLogger(__name__)


class BatchRunner:
    """Handles batch execution of EDG experiments."""

    def __init__(self, batch_config: BatchExperimentConfig):
        """Initialize batch runner.

        Parameters
        ----------
        batch_config : BatchExperimentConfig
            Batch experiment configuration
        """
        self.batch_config = batch_config
        self.results: List[Dict[str, Any]] = []
        self.failed_experiments: List[str] = []
        self.start_time: Optional[float] = None
        self.gpu_manager: Optional[GPUModelManager] = None
        self.results_lock = threading.Lock()
        self.progress_lock = threading.Lock()
        self.completed_count = 0

    def run_batch(self) -> Dict[str, Any]:
        """Run all experiments in the batch.

        Returns
        -------
        Dict[str, Any]
            Batch execution results and summary
        """
        logger.info(f"Starting batch: {self.batch_config.name}")
        self.start_time = time.time()

        # Get all experiment configurations
        experiment_configs = self.batch_config.get_experiment_configs()
        total_experiments = len(experiment_configs)

        if total_experiments == 0:
            logger.warning("No experiments found in batch configuration")
            return self._create_summary()

        logger.info(f"Found {total_experiments} experiments to run")

        # Create batch output directory
        batch_output_dir = Path(self.batch_config.output_base_dir)
        batch_output_dir.mkdir(parents=True, exist_ok=True)

        # Save batch configuration
        batch_config_path = batch_output_dir / "batch_config.yaml"
        from edg.config import save_batch_config

        save_batch_config(self.batch_config, batch_config_path)
        logger.info(f"Saved batch configuration to {batch_config_path}")

        # Initialize progress tracking
        progress_file = batch_output_dir / "progress.json"
        self._save_progress(progress_file, 0, total_experiments)

        try:
            # Choose execution strategy based on max_parallel
            if self.batch_config.max_parallel <= 1:
                # Sequential execution (backward compatibility)
                self._run_sequential(
                    experiment_configs, total_experiments, progress_file
                )
            else:
                # Parallel execution with GPU model manager
                self._run_parallel(experiment_configs, total_experiments, progress_file)
        finally:
            # Clean up GPU models
            if self.gpu_manager:
                self.gpu_manager.cleanup()

        # Create final summary
        summary = self._create_summary()

        # Save summary
        summary_path = batch_output_dir / "batch_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(
            f"Batch completed: {summary['completed_experiments']}/{summary['total_experiments']} successful"
        )
        if summary["failed_experiments"]:
            logger.warning(
                f"Failed experiments: {', '.join(summary['failed_experiments'])}"
            )

        return summary

    def _run_sequential(
        self,
        experiment_configs: List[ExperimentConfig],
        total_experiments: int,
        progress_file: Path,
    ):
        """Run experiments sequentially (original behavior).

        Parameters
        ----------
        experiment_configs : List[ExperimentConfig]
            List of experiment configurations
        total_experiments : int
            Total number of experiments
        progress_file : Path
            Path to progress file
        """
        logger.info("Running experiments sequentially")

        for i, experiment_config in enumerate(experiment_configs):
            try:
                logger.info(
                    f"Running experiment {i + 1}/{total_experiments}: {experiment_config.name}"
                )

                # Run single experiment
                experiment_result = run_experiment(experiment_config)

                # Store result
                self.results.append(
                    {
                        "experiment_name": experiment_config.name,
                        "status": "success",
                        "result": experiment_result,
                    }
                )

                logger.info(
                    f"Completed experiment {i + 1}/{total_experiments}: {experiment_config.name}"
                )

            except Exception as e:
                self._handle_experiment_error(experiment_config, e)

                # Check if we should continue
                if not self.batch_config.continue_on_error:
                    logger.error(
                        "Stopping batch execution due to error (continue_on_error=False)"
                    )
                    break
                else:
                    logger.info(
                        "Continuing with next experiment (continue_on_error=True)"
                    )

            # Update progress
            self._save_progress(progress_file, i + 1, total_experiments)

    def _run_parallel(
        self,
        experiment_configs: List[ExperimentConfig],
        total_experiments: int,
        progress_file: Path,
    ):
        """Run experiments in parallel across multiple GPUs.

        Parameters
        ----------
        experiment_configs : List[ExperimentConfig]
            List of experiment configurations
        total_experiments : int
            Total number of experiments
        progress_file : Path
            Path to progress file
        """
        logger.info(
            f"Running experiments in parallel (max_parallel={self.batch_config.max_parallel})"
        )

        # Initialize GPU model manager
        self.gpu_manager = GPUModelManager(self.batch_config.max_parallel)

        # Pre-load models on all GPUs
        success = self._preload_models(
            experiment_configs[0]
        )  # Use first config for model settings
        if not success:
            logger.error(
                "Failed to pre-load models, falling back to sequential execution"
            )
            self._run_sequential(experiment_configs, total_experiments, progress_file)
            return

        # Run experiments with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.batch_config.max_parallel) as executor:
            # Submit all experiments
            future_to_config = {
                executor.submit(self._run_single_experiment_parallel, config): config
                for config in experiment_configs
            }

            # Process completed experiments
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result()

                    with self.results_lock:
                        self.results.append(
                            {
                                "experiment_name": config.name,
                                "status": "success",
                                "result": result,
                            }
                        )
                        self.completed_count += 1
                        completed = self.completed_count

                    logger.info(
                        f"Completed experiment {completed}/{total_experiments}: {config.name}"
                    )

                except Exception as e:
                    self._handle_experiment_error(config, e)

                    with self.results_lock:
                        self.completed_count += 1
                        completed = self.completed_count

                    # Check if we should continue
                    if not self.batch_config.continue_on_error:
                        logger.error(
                            "Stopping batch execution due to error (continue_on_error=False)"
                        )
                        # Cancel remaining futures
                        for f in future_to_config:
                            f.cancel()
                        break
                    else:
                        logger.info(
                            "Continuing with remaining experiments (continue_on_error=True)"
                        )

                # Update progress
                self._save_progress(progress_file, completed, total_experiments)

    def _preload_models(self, sample_config: ExperimentConfig) -> bool:
        """Pre-load models on all GPUs using sample configuration.

        Parameters
        ----------
        sample_config : ExperimentConfig
            Sample configuration to extract model settings

        Returns
        -------
        bool
            True if all models loaded successfully
        """
        logger.info("Pre-loading models on all GPUs")

        # Extract model settings from sample configuration
        model_version = sample_config.model.version

        # Set checkpoint path
        checkpoint_path = sample_config.model.checkpoint_path
        if checkpoint_path is None:
            if model_version == "boltz1":
                checkpoint_path = str(Path("~/.boltz/boltz1_conf.ckpt").expanduser())
            else:
                checkpoint_path = str(Path("~/.boltz/boltz2_conf.ckpt").expanduser())

        # Set CCD path
        ccd_path = sample_config.model.ccd_path
        if ccd_path is None:
            ccd_path = str(Path("~/.boltz/ccd.pkl").expanduser())

        # Pre-load models
        return self.gpu_manager.preload_models(
            model_version=model_version,
            checkpoint_path=Path(checkpoint_path),
            ccd_path=Path(ccd_path),
            step_scale=sample_config.diffusion.step_scale,
            steering_args=sample_config.steering,  # Pass steering config
        )

    def _run_single_experiment_parallel(self, config: ExperimentConfig) -> Any:
        """Run a single experiment with GPU assignment.

        Parameters
        ----------
        config : ExperimentConfig
            Experiment configuration

        Returns
        -------
        Any
            Experiment result
        """
        device = None
        model = None

        try:
            # Get GPU and model from manager
            device, model = self.gpu_manager.get_gpu_and_model(timeout=30.0)

            # Create modified config with assigned device and model
            modified_config = self._create_config_with_device_and_model(
                config, device, model
            )

            # Run experiment
            logger.info(f"Running experiment {config.name} on {device}")
            result = run_experiment(modified_config)

            return result

        except Exception as e:
            # Handle CUDA OOM specifically
            if (
                "out of memory" in str(e).lower()
                and device
                and device.startswith("cuda")
            ):
                logger.warning(
                    f"CUDA OOM error on {device} for experiment {config.name}"
                )

                # Try to reload model on this GPU for NEXT experiment
                if self.gpu_manager.reload_model_on_gpu(
                    device=device,
                    model_version=config.model.version,
                    checkpoint_path=Path(
                        config.model.checkpoint_path or "~/.boltz/boltz1_conf.ckpt"
                    ),
                    ccd_path=Path(config.model.ccd_path or "~/.boltz/ccd.pkl"),
                    step_scale=config.diffusion.step_scale,
                    steering_args=config.steering,
                ):
                    logger.info(
                        f"Successfully reloaded model on {device} for next experiment"
                    )
                else:
                    logger.error(
                        f"Failed to reload model on {device} - may affect next experiment"
                    )

            # Always re-raise the exception (no retry)
            raise

        finally:
            # Always release GPU back to pool
            if device:
                self.gpu_manager.release_gpu(device)

    def _create_config_with_device_and_model(
        self, config: ExperimentConfig, device: str, model
    ) -> ExperimentConfig:
        """Create a modified config with specific device and model.

        Parameters
        ----------
        config : ExperimentConfig
            Original configuration
        device : str
            Device to assign
        model : Union[Boltz1, Boltz2]
            Pre-loaded model

        Returns
        -------
        ExperimentConfig
            Modified configuration with device and model set
        """
        # Create a copy of the config
        import copy

        modified_config = copy.deepcopy(config)

        # Set device and model
        modified_config.model.device = device
        modified_config.model.pre_loaded_model = model

        return modified_config

    def _handle_experiment_error(self, config: ExperimentConfig, error: Exception):
        """Handle experiment error with proper logging and storage.

        Parameters
        ----------
        config : ExperimentConfig
            Failed experiment configuration
        error : Exception
            The error that occurred
        """
        error_msg = f"Experiment {config.name} failed: {str(error)}"
        logger.error(error_msg)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Full traceback:\n{traceback.format_exc()}")

        # Store failure
        with self.results_lock:
            self.results.append(
                {
                    "experiment_name": config.name,
                    "status": "failed",
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
            self.failed_experiments.append(config.name)

    def _save_progress(self, progress_file: Path, completed: int, total: int):
        """Save progress to file (thread-safe).

        Parameters
        ----------
        progress_file : Path
            Path to progress file
        completed : int
            Number of completed experiments
        total : int
            Total number of experiments
        """
        with self.progress_lock:
            progress_data = {
                "batch_name": self.batch_config.name,
                "completed": completed,
                "total": total,
                "percentage": (completed / total * 100) if total > 0 else 0,
                "timestamp": time.time(),
                "elapsed_time": time.time() - self.start_time if self.start_time else 0,
            }

            if self.gpu_manager:
                progress_data["gpu_models"] = self.gpu_manager.get_model_info()

            with open(progress_file, "w") as f:
                json.dump(progress_data, f, indent=2)

    def _create_summary(self) -> Dict[str, Any]:
        """Create batch execution summary.

        Returns
        -------
        Dict[str, Any]
            Summary of batch execution
        """
        total_experiments = len(self.results)
        successful_experiments = sum(
            1 for r in self.results if r["status"] == "success"
        )
        failed_experiments = sum(1 for r in self.results if r["status"] == "failed")

        elapsed_time = time.time() - self.start_time if self.start_time else 0

        summary = {
            "batch_name": self.batch_config.name,
            "max_parallel": self.batch_config.max_parallel,
            "total_experiments": total_experiments,
            "completed_experiments": successful_experiments,
            "failed_experiments": failed_experiments,
            "success_rate": (successful_experiments / total_experiments * 100)
            if total_experiments > 0
            else 0,
            "failed_experiment_names": self.failed_experiments,
            "elapsed_time_seconds": elapsed_time,
            "results": self.results,
        }

        if self.gpu_manager:
            summary["gpu_models"] = self.gpu_manager.get_model_info()

        return summary


def run_batch_experiment(batch_config: BatchExperimentConfig) -> Dict[str, Any]:
    """Run a batch of EDG experiments.

    Parameters
    ----------
    batch_config : BatchExperimentConfig
        Batch experiment configuration

    Returns
    -------
    Dict[str, Any]
        Batch execution results and summary
    """
    runner = BatchRunner(batch_config)
    return runner.run_batch()
