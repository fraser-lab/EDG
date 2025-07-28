"""GPU Model Manager for multi-GPU parallel execution.

This module manages Boltz model loading and assignment across multiple GPUs
for efficient batch processing without redundant model loading.
"""

import logging
import queue
import threading
import time
import random
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
        from boltz.main import (
            BoltzDiffusionParams,
            Boltz2DiffusionParams,
            BoltzSteeringParams,
            PairformerArgs,
            PairformerArgsV2,
            MSAModuleArgs,
        )
        from edg.edg.modules.diffusion import PredictArgs
        from dataclasses import asdict

        # Create default parameters (similar to DiffusionStepper)
        predict_args = PredictArgs()

        # Set args based on model version
        if model_version == "boltz1":
            pairformer_args = PairformerArgs()
        else:  # boltz2
            pairformer_args = PairformerArgsV2()

        # MSAModuleArgs with correct parameters based on boltz repo
        msa_args = MSAModuleArgs(
            subsample_msa=True,
            num_subsampled_msa=1024,
            use_paired_feature=(model_version == "boltz2"),
        )

        # Create proper BoltzSteeringParams from config (like optimizer_factory.py does)
        steering_config = model_kwargs.get("steering_args", None)
        steering_args = BoltzSteeringParams()
        if steering_config is not None:
            # Convert config object to BoltzSteeringParams
            steering_args.fk_steering = steering_config.enabled
            steering_args.physical_guidance_update = steering_config.physical_guidance_update
            steering_args.num_particles = steering_config.num_particles
            steering_args.contact_guidance_update = steering_config.contact_guidance_update

        # Create diffusion args based on model version
        if model_version == "boltz1":
            diffusion_args = BoltzDiffusionParams(
                step_scale=model_kwargs.get("step_scale") or 1.638
            )
        else:  # boltz2
            diffusion_args = Boltz2DiffusionParams(
                step_scale=model_kwargs.get("step_scale") or 1.5
            )

        # Load model
        if model_version == "boltz1":
            model = Boltz1.load_from_checkpoint(
                checkpoint_path,
                strict=True,
                predict_args=asdict(predict_args),  # Convert dataclass to dict
                map_location="cpu",
                diffusion_process_args=asdict(diffusion_args),
                ema=False,
                use_kernels=True,  # Required parameter for Boltz1
                pairformer_args=asdict(pairformer_args),
                msa_args=asdict(msa_args),  # Correct parameter name
                steering_args=asdict(steering_args),
            )
        else:  # boltz2
            model = Boltz2.load_from_checkpoint(
                checkpoint_path,
                strict=True,
                predict_args=asdict(predict_args),  # Convert dataclass to dict
                map_location="cpu",
                diffusion_process_args=asdict(diffusion_args),
                ema=False,
                pairformer_args=asdict(pairformer_args),
                msa_args=asdict(msa_args),  # Correct parameter name
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
        RuntimeError
            If no model loaded on acquired device
        """
        device = self._available_gpus.get(timeout=timeout)

        with self._lock:
            if device not in self._gpu_models:
                # Return device to queue before raising exception
                self._available_gpus.put(device)
                raise RuntimeError(f"No model loaded on device {device}")
            model = self._gpu_models[device]
            
            # Validate model is not None
            if model is None:
                self._available_gpus.put(device)
                raise RuntimeError(f"Model on device {device} is None")

        logger.debug(f"Assigned {device}")
        return device, model

    def get_gpu_and_model_with_retry(
        self, 
        base_timeout: float = 30.0, 
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        jitter: bool = True
    ) -> Tuple[str, Union[Boltz1, Boltz2]]:
        """Get GPU with exponential backoff retry logic.

        Parameters
        ----------
        base_timeout : float
            Initial timeout in seconds for GPU acquisition
        max_retries : int
            Maximum number of retry attempts
        backoff_factor : float
            Multiplier for timeout on each retry
        jitter : bool
            Add random jitter to prevent thundering herd

        Returns
        -------
        Tuple[str, Union[Boltz1, Boltz2]]
            Device ID and loaded model instance

        Raises
        ------
        RuntimeError
            If all retry attempts fail or model validation fails
        """
        last_exception = None
        
        for attempt in range(max_retries + 1):
            # Calculate timeout with exponential backoff
            timeout = base_timeout * (backoff_factor ** attempt)
            
            # Add jitter to prevent thundering herd effect
            if jitter and attempt > 0:
                jitter_range = min(5.0, timeout * 0.1)
                timeout += random.uniform(0, jitter_range)
            
            try:
                logger.debug(f"GPU acquisition attempt {attempt + 1}/{max_retries + 1}, timeout={timeout:.1f}s")
                
                # Try to get GPU from queue
                device = self._available_gpus.get(timeout=timeout)
                
                # Validate GPU and model health
                if self._validate_gpu_and_model(device):
                    with self._lock:
                        model = self._gpu_models[device]
                    
                    logger.debug(f"Assigned {device}")
                    return device, model
                else:
                    # GPU/model validation failed, return to queue and try again
                    logger.warning(f"GPU {device} failed validation, returning to queue")
                    self._available_gpus.put(device)
                    
                    if attempt < max_retries:
                        continue
                    else:
                        raise RuntimeError(f"GPU {device} repeatedly failed validation")
                        
            except queue.Empty as e:
                last_exception = e
                logger.warning(f"GPU queue timeout after {timeout:.1f}s (attempt {attempt + 1}/{max_retries + 1})")
                
                if attempt < max_retries:
                    # Wait before retrying with jitter
                    delay = 2.0 + (3.0 * attempt)
                    if jitter:
                        delay += random.uniform(0, 2.0)
                    
                    logger.debug(f"Waiting {delay:.1f}s before retry...")
                    time.sleep(delay)
                    continue
            
            except Exception as e:
                logger.error(f"Unexpected error in GPU acquisition attempt {attempt + 1}: {e}")
                last_exception = e
                if attempt < max_retries:
                    time.sleep(1.0)
                    continue
        
        # All retries failed
        queue_size = self._available_gpus.qsize()
        total_gpus = len(self._gpu_list)
        busy_gpus = total_gpus - queue_size
        
        error_msg = (
            f"Failed to acquire GPU after {max_retries + 1} attempts. "
            f"Queue state: {queue_size}/{total_gpus} available, {busy_gpus} busy. "
            f"Last error: {last_exception}"
        )
        
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    def _validate_gpu_and_model(self, device: str) -> bool:
        """Validate GPU health and model functionality.

        Parameters
        ----------
        device : str
            Device ID to validate

        Returns
        -------
        bool
            True if GPU and model are healthy, False otherwise
        """
        try:
            with self._lock:
                # Check if model exists for this device
                if device not in self._gpu_models:
                    logger.error(f"No model loaded on device {device}")
                    return False
                
                model = self._gpu_models[device]
                if model is None:
                    logger.error(f"Model on device {device} is None")
                    return False
            
            # Check GPU memory health (if CUDA device)
            if device.startswith("cuda"):
                try:
                    device_obj = torch.device(device)
                    
                    # Check if device is accessible
                    torch.cuda.set_device(device_obj)
                    
                    # Check available memory
                    memory_allocated = torch.cuda.memory_allocated(device_obj)
                    memory_total = torch.cuda.get_device_properties(device_obj).total_memory
                    memory_usage = memory_allocated / memory_total
                    
                    logger.debug(f"GPU {device} memory usage: {memory_usage:.1%} ({memory_allocated / 1024**3:.1f}GB / {memory_total / 1024**3:.1f}GB)")
                    
                    # Warn if memory usage is very high (>95%)
                    if memory_usage > 0.95:
                        logger.warning(f"GPU {device} has high memory usage: {memory_usage:.1%}")
                        # Still allow usage but log warning
                    
                    # Test basic GPU operations
                    test_tensor = torch.randn(10, 10, device=device_obj)
                    test_result = test_tensor.sum()
                    del test_tensor, test_result
                    
                except Exception as e:
                    logger.error(f"GPU {device} failed health check: {e}")
                    return False
            
            # Basic model validation - check if model has required attributes
            if not hasattr(model, 'device') and not hasattr(model, 'to'):
                logger.error(f"Model on device {device} appears to be corrupted (missing basic attributes)")
                return False
            
            logger.debug(f"GPU {device} and model passed validation")
            return True
            
        except Exception as e:
            logger.error(f"Unexpected error validating GPU {device}: {e}")
            return False

    def get_queue_stats(self) -> Dict[str, Union[int, float]]:
        """Get current queue statistics for monitoring.

        Returns
        -------
        Dict[str, Union[int, float]]
            Queue statistics including size, total GPUs, utilization
        """
        queue_size = self._available_gpus.qsize()
        total_gpus = len(self._gpu_list)
        busy_gpus = total_gpus - queue_size
        utilization = busy_gpus / total_gpus if total_gpus > 0 else 0.0
        
        return {
            'queue_size': queue_size,
            'total_gpus': total_gpus,
            'busy_gpus': busy_gpus,
            'utilization': utilization
        }

    def release_gpu(self, device: str):
        """Release GPU back to available pool.

        Parameters
        ----------
        device : str
            Device ID to release
        """
        logger.debug(f"Released {device}")
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
