"""GPU Model Manager for multi-GPU parallel execution.

This module manages Boltz model loading and assignment across multiple GPUs
for efficient batch processing without redundant model loading.
"""

import logging
import queue
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List
import torch
import gc

from boltz.model.models.boltz1 import Boltz1
from boltz.model.models.boltz2 import Boltz2

logger = logging.getLogger(__name__)


class GPUModelManager:
    """Manages Boltz model loading and assignment across multiple GPUs."""

    def __init__(self, max_parallel: int):
        """Initialize GPU model manager.

        Parameters
        ----------
        max_parallel : int
            Maximum number of parallel experiments (and GPUs to use)
        """
        self.max_parallel = max_parallel
        self._gpu_models: Dict[str, Union[Boltz1, Boltz2]] = {}
        self._available_gpus = queue.Queue()
        self._lock = threading.Lock()
        self._gpu_list: List[str] = []

        # Initialize available GPUs
        self._initialize_gpu_list()

    def _initialize_gpu_list(self):
        """Initialize list of available GPUs."""
        try:
            gpu_count = torch.cuda.device_count()
            if gpu_count == 0:
                logger.warning("No CUDA GPUs available, falling back to CPU")
                self._gpu_list = ["cpu"]
            else:
                # Use up to max_parallel GPUs
                num_gpus_to_use = min(self.max_parallel, gpu_count)
                self._gpu_list = [f"cuda:{i}" for i in range(num_gpus_to_use)]
                logger.info(f"Using {num_gpus_to_use} GPUs: {self._gpu_list}")
        except Exception as e:
            logger.error(f"Error initializing GPU list: {e}")
            self._gpu_list = ["cpu"]

        # Fill available GPU queue
        for gpu in self._gpu_list:
            self._available_gpus.put(gpu)

    def preload_models(
        self, model_version: str, checkpoint_path: Path, ccd_path: Path, **model_kwargs
    ) -> bool:
        """Pre-load models onto all available GPUs.

        Parameters
        ----------
        model_version : str
            Model version ("boltz1" or "boltz2")
        checkpoint_path : Path
            Path to model checkpoint
        ccd_path : Path
            Path to CCD file
        **model_kwargs
            Additional keyword arguments for model loading

        Returns
        -------
        bool
            True if all models loaded successfully, False otherwise
        """
        logger.info(
            f"Pre-loading {model_version} models onto {len(self._gpu_list)} GPUs"
        )

        success = True
        for device in self._gpu_list:
            try:
                model = self._load_model_on_device(
                    model_version, checkpoint_path, ccd_path, device, **model_kwargs
                )
                self._gpu_models[device] = model
                logger.info(f"Successfully loaded {model_version} on {device}")
            except Exception as e:
                logger.error(f"Failed to load {model_version} on {device}: {e}")
                success = False

        return success

    def _load_model_on_device(
        self,
        model_version: str,
        checkpoint_path: Path,
        ccd_path: Path,
        device: str,
        **model_kwargs,
    ) -> Union[Boltz1, Boltz2]:
        """Load model on specific device.

        Parameters
        ----------
        model_version : str
            Model version ("boltz1" or "boltz2")
        checkpoint_path : Path
            Path to model checkpoint
        ccd_path : Path
            Path to CCD file
        device : str
            Device to load model on
        **model_kwargs
            Additional keyword arguments for model loading

        Returns
        -------
        Union[Boltz1, Boltz2]
            Loaded model instance
        """
        logger.debug(f"Loading {model_version} model on {device}")

        # Import here to avoid circular imports
        from boltz.main import BoltzDiffusionParams, Boltz2DiffusionParams
        from boltz.main import (
            BoltzPairformerParams,
            BoltzMSAParams,
            BoltzSteeringParams,
        )
        from boltz.main import BoltzPredictParams
        from dataclasses import asdict

        # Create default parameters (similar to DiffusionStepper)
        predict_args = BoltzPredictParams()
        pairformer_args = BoltzPairformerParams()
        msa_args = BoltzMSAParams()
        steering_args = model_kwargs.get("steering_args", BoltzSteeringParams())

        # Create diffusion args based on model version
        if model_version == "boltz1":
            diffusion_args = BoltzDiffusionParams(
                step_scale=model_kwargs.get("step_scale", 1.638)
            )
        else:  # boltz2
            diffusion_args = Boltz2DiffusionParams(
                step_scale=model_kwargs.get("step_scale", 1.5)
            )

        # Load model
        if model_version == "boltz1":
            model = Boltz1.load_from_checkpoint(
                checkpoint_path,
                strict=True,
                predict_args=asdict(predict_args),
                map_location="cpu",
                diffusion_process_args=asdict(diffusion_args),
                ema=False,
                use_kernels=True,
                pairformer_args=asdict(pairformer_args),
                msa_args=asdict(msa_args),
                steering_args=asdict(steering_args),
            )
        else:  # boltz2
            model = Boltz2.load_from_checkpoint(
                checkpoint_path,
                strict=True,
                predict_args=asdict(predict_args),
                map_location="cpu",
                diffusion_process_args=asdict(diffusion_args),
                ema=False,
                pairformer_args=asdict(pairformer_args),
                msa_args=asdict(msa_args),
                steering_args=asdict(steering_args),
            )

        # Move to device and set to eval mode
        model = model.to(device).eval()

        return model

    def get_gpu_and_model(
        self, timeout: Optional[float] = None
    ) -> Tuple[str, Union[Boltz1, Boltz2]]:
        """Get next available GPU with loaded model.

        Parameters
        ----------
        timeout : Optional[float]
            Timeout in seconds to wait for available GPU

        Returns
        -------
        Tuple[str, Union[Boltz1, Boltz2]]
            Device ID and loaded model instance

        Raises
        ------
        queue.Empty
            If no GPU becomes available within timeout
        """
        device = self._available_gpus.get(timeout=timeout)

        with self._lock:
            if device not in self._gpu_models:
                raise RuntimeError(f"No model loaded on device {device}")
            model = self._gpu_models[device]

        logger.debug(f"Assigned {device} with model to experiment")
        return device, model

    def release_gpu(self, device: str):
        """Release GPU back to available pool.

        Parameters
        ----------
        device : str
            Device ID to release
        """
        logger.debug(f"Releasing {device} back to pool")
        self._available_gpus.put(device)

    def reload_model_on_gpu(
        self,
        device: str,
        model_version: str,
        checkpoint_path: Path,
        ccd_path: Path,
        **model_kwargs,
    ) -> bool:
        """Reload model on GPU after OOM error.

        Parameters
        ----------
        device : str
            Device to reload model on
        model_version : str
            Model version ("boltz1" or "boltz2")
        checkpoint_path : Path
            Path to model checkpoint
        ccd_path : Path
            Path to CCD file
        **model_kwargs
            Additional keyword arguments for model loading

        Returns
        -------
        bool
            True if model reloaded successfully, False otherwise
        """
        logger.info(f"Reloading model on {device} after error")

        with self._lock:
            # Clear GPU memory
            if device in self._gpu_models:
                del self._gpu_models[device]

            if device.startswith("cuda"):
                torch.cuda.empty_cache()
                gc.collect()

        try:
            # Reload model
            model = self._load_model_on_device(
                model_version, checkpoint_path, ccd_path, device, **model_kwargs
            )

            with self._lock:
                self._gpu_models[device] = model

            logger.info(f"Successfully reloaded model on {device}")
            return True

        except Exception as e:
            logger.error(f"Failed to reload model on {device}: {e}")
            return False

    def cleanup(self):
        """Clean up all GPU models and free memory."""
        logger.info("Cleaning up GPU models")

        with self._lock:
            for device in list(self._gpu_models.keys()):
                del self._gpu_models[device]
            self._gpu_models.clear()

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def get_model_info(self) -> Dict[str, str]:
        """Get information about loaded models.

        Returns
        -------
        Dict[str, str]
            Dictionary mapping device to model type
        """
        with self._lock:
            return {
                device: type(model).__name__
                for device, model in self._gpu_models.items()
            }
